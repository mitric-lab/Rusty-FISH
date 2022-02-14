use crate::constants;
use crate::initialization::PulseConfiguration;
use crate::initialization::Simulation;
use ndarray::prelude::*;
use ndarray::{array, Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::Lapack;
use ndarray_linalg::Scalar;
use ndarray_linalg::{c64, eigh, into_col, into_row, solve, Eig, Eigh, Inverse, Solve, UPLO};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::fs;
use std::path::Path;
use toml;

impl Simulation {
    pub fn get_hopping_fortran(&self, actual_step: f64) -> Array1<c64> {
        // Initialization
        let old_nonadiabatic_scalar: Array2<f64> = -1.0 * &self.nonadiabatic_scalar_old;
        let nonadiabatic_scalar: Array2<f64> = -1.0 * &self.nonadiabatic_scalar;

        let n_delta: usize = self.config.hopping_config.integration_steps;
        let delta_rk: f64 = self.stepsize / n_delta as f64;
        let coupling_flag: i8 = self.config.hopping_config.coupling_flag;

        let mut nonadibatic_slope: Array2<f64> = Array2::zeros(nonadiabatic_scalar.raw_dim());
        if coupling_flag == 1 || coupling_flag == 2 {
            for i in (0..self.config.nstates) {
                for j in (0..self.config.nstates) {
                    nonadibatic_slope[[i, j]] = (nonadiabatic_scalar[[i, j]]
                        - old_nonadiabatic_scalar[[i, j]])
                        / (n_delta as f64 * delta_rk);
                }
            }
        }

        // Get the electric field of the pulse
        let mut electric_field: Array1<f64> = Array1::zeros(4 * n_delta);
        if coupling_flag == 0 || coupling_flag == 2 {
            electric_field = get_analytic_field(
                &self.config.pulse_config,
                4 * n_delta,
                delta_rk,
                actual_step,
            );
        }
        // println!("electric field {}",electric_field.slice(s![0..10]));

        // Integration
        let t_start: f64 = delta_rk * actual_step;
        let mut t_i: f64 = 0.0;
        let mut old_coefficients: Array1<c64> = self.coefficients.clone();

        for i in (0..n_delta) {
            t_i = i as f64 * delta_rk;
            let t_abs: f64 = t_start + t_i;

            // do runge kutta
            let new_coefficients: Array1<c64> = runge_kutta_integration(
                i + 1,
                t_i,
                old_coefficients.view(),
                self.energies.view(),
                actual_step,
                self.config.nstates,
                self.state,
                n_delta,
                delta_rk,
                self.dipole.view(),
                old_nonadiabatic_scalar.view(),
                nonadibatic_slope.view(),
                coupling_flag,
                electric_field.view(),
            );
            old_coefficients = new_coefficients;
        }
        let time: f64 = delta_rk * n_delta as f64;
        let energy_compl: Array1<c64> = self
            .energies
            .mapv(|val| (-c64::new(0.0, 1.0) * val * time).exp());
        let c_new: Array1<c64> = old_coefficients * energy_compl;

        return c_new;
    }

    // The coefficients of the electronic wavefunction are propagated
    // in the local diabatic basis as explained in
    // [1]  JCP 114, 10608 (2001) and
    // [2]  JCP 137, 22A514 (2012)
    pub fn get_local_diabatization(
        &self,
        energy_last: ArrayView1<f64>,
        t_tot_last: Option<Array2<f64>>,
    ) -> (Array1<c64>, Array2<f64>, Array2<f64>) {
        // Loewding orthogonalization of the S matrix
        // see eqns. (B5) and (B6) in [2]

        let s_ts: Array2<f64> = self.s_mat.t().dot(&self.s_mat);
        let (l, o): (Array1<f64>, Array2<f64>) = s_ts.eigh(UPLO::Upper).unwrap();
        let lm12: Array1<f64> = (1.0 / l).mapv(|val| val.sqrt());

        // unitary transformation matrix, see eqn. (B5) in [1]
        let t: Array2<f64> = self.s_mat.dot(&o.dot(&Array::from_diag(&lm12).dot(&o.t())));

        let t_inv: Array2<f64> = t.clone().reversed_axes();
        // electronic coefficients c(t)
        let c_0: ArrayView1<c64> = self.coefficients.view();
        // adiabatic energies at the beginning of the time step, E(t)
        let e_0: ArrayView1<f64> = energy_last;
        // adiabatic energies at the end of the time step, E(t+dt)
        let e_1: ArrayView1<f64> = self.energies.view();

        // diabatic hamiltonian H(t+dt)
        let h: Array2<f64> = t.dot(&Array::from_diag(&e_1).dot(&t_inv));
        let mut h_interp: Array2<f64> = (Array::from_diag(&e_0) + h) / 2.0;

        // subtract lowest energy from diagonal
        let h_00_val: f64 = h_interp[[0, 0]];
        for ii in (0..h_interp.dim().0) {
            h_interp[[ii, ii]] -= h_00_val;
        }

        // propagator in diabatic basis, see eqn. (11) in [1]
        let u_1: Array2<c64> = h_interp.mapv(|val| -c64::new(0.0, 1.0) * val * self.stepsize);
        let (eig, eig_vec): (Array1<c64>, Array2<c64>) = u_1.eig().unwrap();
        let diag: Array1<c64> = eig.mapv(|val| val.exp());
        let u_mat: Array2<c64> = eig_vec.dot(&Array::from_diag(&diag).dot(&eig_vec.inv().unwrap()));
        // let u: Array2<c64> =
        //     h_interp.mapv(|val| (-c64::new(0.0, 1.0) * val* stepsize).exp());
        // println!("U {}",u);

        // at the beginning of the time step the adiabatic and diabatic basis is assumed to coincide
        // new electronic coefficients c(t+dt) in the adiabatic basis
        let complex_t_inv: Array2<c64> = t_inv.mapv(|val| val * c64::new(1.0, 0.0));
        // let c_1: Array1<c64> = u_mat.dot(&c_0).dot(&complex_t_inv.t());
        let c_1: Array1<c64> = complex_t_inv.t().dot(&u_mat.dot(&c_0));

        // norm of electronic wavefunction
        let norm_c: f64 = c_1.map(|val| val.re.powi(2) + val.im.powi(2)).sum();
        assert!(
            norm_c - 1.0 < 1.0e-3,
            "Norm of electronic coefficients not conserved! Norm = {}",
            norm_c
        );

        // save the diabatic hamiltonian along the trajectory
        let mut ttot_last: Array2<f64> = Array2::zeros(t.raw_dim());
        if t_tot_last.is_none() {
            ttot_last = Array::eye(t.dim().0);
        }

        // the transformations are concatenated to obtain the diabatic
        // Hamiltonian relative to the first time step
        let t_tot: Array2<f64> = t.dot(&ttot_last);
        let t_tot_inv: Array2<f64> = t_tot.clone().reversed_axes();

        let h_diab: Array2<f64> = t_tot.dot(&Array::from_diag(&e_1).dot(&t_tot_inv));
        ttot_last = t_tot;

        return (c_1, h_diab, ttot_last);
    }
}

pub fn runge_kutta_integration(
    iterator: usize,
    time: f64,
    coefficients: ArrayView1<c64>,
    energy: ArrayView1<f64>,
    actual_step: f64,
    nstates: usize,
    actual_state: usize,
    n_delta: usize,
    delta_rk: f64,
    dipole: ArrayView3<f64>,
    old_nonadiabatic_scalar: ArrayView2<f64>,
    nonadiabatic_slope: ArrayView2<f64>,
    coupling_flag: i8,
    efield: ArrayView1<f64>,
) -> Array1<c64> {
    let n: usize = 4 * iterator - 3;
    let mut k_1: Array1<c64> = runge_kutta_helper(
        time,
        coefficients,
        energy,
        nstates,
        actual_state,
        n_delta,
        n,
        dipole,
        old_nonadiabatic_scalar,
        nonadiabatic_slope,
        coupling_flag,
        efield,
    );
    k_1 = k_1 * delta_rk;
    let tmp: Array1<c64> = &coefficients + &(&k_1 * 0.5);

    let n: usize = 4 * iterator - 2;
    let mut k_2: Array1<c64> = runge_kutta_helper(
        time + 0.5 * delta_rk,
        tmp.view(),
        energy,
        nstates,
        actual_state,
        n_delta,
        n,
        dipole,
        old_nonadiabatic_scalar,
        nonadiabatic_slope,
        coupling_flag,
        efield,
    );
    k_2 = k_2 * delta_rk;
    let tmp: Array1<c64> = &coefficients + &(&k_2 * 0.5);

    let n: usize = 4 * iterator - 1;
    let mut k_3: Array1<c64> = runge_kutta_helper(
        time + 0.5 * delta_rk,
        tmp.view(),
        energy,
        nstates,
        actual_state,
        n_delta,
        n,
        dipole,
        old_nonadiabatic_scalar,
        nonadiabatic_slope,
        coupling_flag,
        efield,
    );
    k_3 = k_3 * delta_rk;
    let tmp: Array1<c64> = &coefficients + &(&k_3 * 0.5);

    let n: usize = 4 * iterator;
    let mut k_4: Array1<c64> = runge_kutta_helper(
        time + delta_rk,
        tmp.view(),
        energy,
        nstates,
        actual_state,
        n_delta,
        n,
        dipole,
        old_nonadiabatic_scalar,
        nonadiabatic_slope,
        coupling_flag,
        efield,
    );
    k_4 = k_4 * delta_rk;

    let new_coefficients: Array1<c64> =
        &coefficients + &((k_1 + k_2 * 2.0 + k_3 * 2.0 + k_4) * 1.0 / 6.0);
    return new_coefficients;
}

fn runge_kutta_helper(
    time: f64,
    coefficients: ArrayView1<c64>,
    energy: ArrayView1<f64>,
    nstates: usize,
    actual_state: usize,
    n_delta: usize,
    n: usize,
    dipole: ArrayView3<f64>,
    old_nonadiabatic_scalar: ArrayView2<f64>,
    nonadiabatic_slope: ArrayView2<f64>,
    coupling_flag: i8,
    efield: ArrayView1<f64>,
) -> Array1<c64> {
    let mut f: Array1<c64> = Array1::zeros(coefficients.raw_dim());
    let mut non_adiabatic: Array2<f64> = Array2::zeros(nonadiabatic_slope.raw_dim());
    let mut field_coupling: Array2<f64> = Array2::zeros(nonadiabatic_slope.raw_dim());

    // coupling_flag =0: field coupling only; 1: nonadiabatic coupling only; 2: both
    if coupling_flag == 0 || coupling_flag == 2 {
        field_coupling = get_field_coupling(dipole, n, nstates, efield, n_delta);
    }

    if coupling_flag == 1 || coupling_flag == 2 {
        non_adiabatic =
            get_nonadiabatic_coupling(time, nonadiabatic_slope, old_nonadiabatic_scalar, nstates);
    }
    // create energy difference array
    let mut energy_arr_tmp: Array2<f64> = energy.to_owned().insert_axis(Axis(1));
    let mesh_1: ArrayView2<f64> = energy_arr_tmp.broadcast((nstates, nstates)).unwrap();
    let energy_difference: Array2<f64> = &mesh_1.clone() - &mesh_1.t();

    // println!("nonadiabatic {}",non_adiabatic);
    // println!("coupling {}",field_coupling.t());
    //
    // for k in (0..nstates) {
    //     for l in (0..nstates) {
    //         let mut incr: c64 = c64::new(0.0, 0.0);
    //
    //         if coupling_flag > 0 {
    //             incr = incr - non_adiabatic[[l, k]];
    //         }
    //         println!("incr_0 {}",incr);
    //         if coupling_flag == 0 || coupling_flag == 2 {
    //             incr = incr - c64::new(0.0, 1.0) * field_coupling[[l, k]];
    //         }
    //         println!("incr_1 {}",incr);
    //
    //         f[k] = f[k]
    //             + incr
    //                 * coefficients[l]
    //                 * (c64::new(0.0, 1.0) * (energy[k] - energy[l])* time).exp();
    //     }
    // }

    // alternative way instead of iteration
    let dE: Array2<c64> = energy_difference.mapv(|val| (c64::new(0.0, 1.0) * val * time).exp());
    let mut incr: Array2<f64> = Array2::zeros((nstates, nstates));
    incr = incr + non_adiabatic;
    let incr_complex = field_coupling.mapv(|val| c64::new(0.0, -1.0) * val) + incr;
    // let h:Array2<c64> = dE * non_adiabatic;
    let h: Array2<c64> = dE * incr_complex;
    let f_new: Array1<c64> = h.dot(&coefficients); //.mapv(|val| val *c64::new(0.0,-1.0));

    return f_new;
}

fn get_nonadiabatic_coupling(
    time: f64,
    nonadiabatic_slope: ArrayView2<f64>,
    old_nonadiabatic_scalar: ArrayView2<f64>,
    nstates: usize,
) -> Array2<f64> {
    let mut nonadiabatic: Array2<f64> = Array::zeros(old_nonadiabatic_scalar.raw_dim());
    for i in (0..nstates) {
        for j in (0..nstates) {
            nonadiabatic[[i, j]] =
                old_nonadiabatic_scalar[[i, j]] + nonadiabatic_slope[[i, j]] * time;
        }
    }
    return nonadiabatic;
}

fn get_field_coupling(
    dipole: ArrayView3<f64>,
    n: usize,
    nstates: usize,
    efield: ArrayView1<f64>,
    n_delta: usize,
) -> Array2<f64> {
    let mut coupling: Array2<f64> = Array2::zeros((nstates, nstates));
    let efield_index: usize = n - 1;
    for i in (0..nstates) {
        for j in (i + 1..nstates) {
            coupling[[i, j]] = -1.0 / 3.0_f64.sqrt()
                * (dipole[[i, j, 0]] + dipole[[i, j, 1]] + dipole[[i, j, 2]])
                * efield[efield_index];
            coupling[[j, i]] = coupling[[i, j]]
        }
    }
    return coupling;
}

fn get_electric_field(
    config: &PulseConfiguration,
    nstep: usize,
    tstep: f64,
    nactstep: f64,
) -> Array1<f64> {
    let electric_field: Array1<f64> = get_analytic_field(config, nstep, tstep, nactstep);
    return electric_field;
}

fn get_analytic_field(
    config: &PulseConfiguration,
    nstep: usize,
    tstep: f64,
    nactstep: f64,
) -> Array1<f64> {
    let mut electric_field: Array1<f64> = Array1::zeros(nstep);
    // read pulse parameters from external file
    let e0: f64 = config.e0;
    let omega: f64 = config.omega;
    let alpha: f64 = config.gaussian_factor;
    let t0: f64 = config.time_delay;

    for step in (0..nstep) {
        let time: f64 = tstep * (nactstep + 0.25 * step as f64);
        electric_field[step] +=
            e0 * (omega * (time - t0)).cos() * (-alpha * (time - t0).powi(2)).exp();
    }
    return electric_field;
}

// The coefficients of the electronic wavefunction are propagated
// in the local diabatic basis as explained in
// [1]  JCP 114, 10608 (2001) and
// [2]  JCP 137, 22A514 (2012)
pub fn get_local_diabatization(
    overlap_matrix: ArrayView2<f64>,
    coefficients: ArrayView1<c64>,
    energy_last: ArrayView1<f64>,
    energy: ArrayView1<f64>,
    stepsize: f64,
    t_tot_last: Option<Array2<f64>>,
) -> (Array1<c64>, Array2<f64>, Array2<f64>) {
    // Loewding orthogonalization of the S matrix
    // see eqns. (B5) and (B6) in [2]
    println!("overlap matrix {}", overlap_matrix);
    let overlap_matrix: Array2<f64> = array![[1., 2., 3.], [2., 4., 4.], [3., 4., 5.]];

    let s_ts: Array2<f64> = overlap_matrix.t().dot(&overlap_matrix);
    let (l, o): (Array1<f64>, Array2<f64>) = s_ts.eigh(UPLO::Upper).unwrap();
    println!("L {}", l);
    let lm12: Array1<f64> = (1.0 / l).mapv(|val| val.sqrt());
    println!("lm12 {}", lm12);
    // unitary transformation matrix, see eqn. (B5) in [1]
    let t: Array2<f64> = overlap_matrix.dot(&o.dot(&Array::from_diag(&lm12).dot(&o.t())));
    println!("T {}", t);
    let t_inv: Array2<f64> = t.clone().reversed_axes();
    // electronic coefficients c(t)
    let c_0: Array1<c64> = coefficients.to_owned();
    // adiabatic energies at the beginning of the time step, E(t)
    let e_0: ArrayView1<f64> = energy_last;
    // adiabatic energies at the end of the time step, E(t+dt)
    let e_1: ArrayView1<f64> = energy;
    println!("E0 {}", e_0);
    println!("E1 {}", e_1);

    // diabatic hamiltonian H(t+dt)
    let h: Array2<f64> = t.dot(&Array::from_diag(&e_1).dot(&t_inv));
    println!("H {}", h);
    let mut h_interp: Array2<f64> = (Array::from_diag(&e_0) + h) / 2.0;
    println!("Hinterp {}", h_interp);

    // subtract lowest energy from diagonal
    let h_00_val: f64 = h_interp[[0, 0]];
    for ii in (0..h_interp.dim().0) {
        h_interp[[ii, ii]] -= h_00_val;
    }
    println!("Hinterp v2 {}", h_interp);
    println!("tstep {}", stepsize);
    // propagator in diabatic basis, see eqn. (11) in [1]
    let u_1: Array2<c64> = h_interp.mapv(|val| -c64::new(0.0, 1.0) * val * stepsize);
    let (eig, eig_vec): (Array1<c64>, Array2<c64>) = u_1.eig().unwrap();
    let diag: Array1<c64> = eig.mapv(|val| val.exp());
    let u_mat: Array2<c64> = eig_vec.dot(&Array::from_diag(&diag).dot(&eig_vec.inv().unwrap()));
    // let u: Array2<c64> =
    //     h_interp.mapv(|val| (-c64::new(0.0, 1.0) * val* stepsize).exp());
    // println!("U {}",u);
    println!("U matrix: {}", u_mat);
    // at the beginning of the time step the adiabatic and diabatic basis is assumed to coincide
    // new electronic coefficients c(t+dt) in the adiabatic basis
    let complex_t_inv: Array2<c64> = t_inv.mapv(|val| val * c64::new(1.0, 0.0));
    // let c_1: Array1<c64> = u_mat.dot(&c_0).dot(&complex_t_inv.t());
    let c_1: Array1<c64> = complex_t_inv.t().dot(&u_mat.dot(&c_0));
    println!("coeff {}", c_1);

    // norm of electronic wavefunction
    let norm_c: f64 = c_1.map(|val| val.re.powi(2) + val.im.powi(2)).sum();
    assert!(
        norm_c - 1.0 < 1.0e-3,
        "Norm of electronic coefficients not conserved! Norm = {}",
        norm_c
    );

    // save the diabatic hamiltonian along the trajectory
    let mut ttot_last: Array2<f64> = Array2::zeros(t.raw_dim());
    if t_tot_last.is_none() {
        ttot_last = Array::eye(t.dim().0);
    }

    // the transformations are concatenated to obtain the diabatic
    // Hamiltonian relative to the first time step
    let t_tot: Array2<f64> = t.dot(&ttot_last);
    let t_tot_inv: Array2<f64> = t_tot.clone().reversed_axes();

    let h_diab: Array2<f64> = t_tot.dot(&Array::from_diag(&e_1).dot(&t_tot_inv));
    ttot_last = t_tot;

    return (c_1, h_diab, ttot_last);
}
