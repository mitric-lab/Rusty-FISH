use crate::constants;
use crate::defaults::PULSE_CONFIG;
use crate::initialization::Pulse_Configuration;
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

        let n_delta: usize = self.config.n_small_steps;
        let delta_rk: f64 = self.delta_runge_kutta;
        let coupling_flag: i8 = self.config.coupling;

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
            electric_field =
                get_electric_field(self.config.fieldflag, 4 * n_delta, delta_rk, actual_step);
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

pub fn get_hopping_fortran(
    coefficients: ArrayView1<c64>,
    energy: ArrayView1<f64>,
    actual_step: f64,
    nstates: usize,
    actual_state: usize,
    n_delta: usize,
    delta_rk: f64,
    dipole: ArrayView3<f64>,
    old_nonadiabatic_scalar: ArrayView2<f64>,
    nonadiabatic_scalar: ArrayView2<f64>,
    coupling_flag: i8,
    field_flag: u8,
) -> Array1<c64> {
    // Initialization
    let old_nonadiabatic_scalar: Array2<f64> = -1.0 * &old_nonadiabatic_scalar;
    let nonadiabatic_scalar: Array2<f64> = -1.0 * &nonadiabatic_scalar;

    let mut nonadibatic_slope: Array2<f64> = Array2::zeros(nonadiabatic_scalar.raw_dim());
    if coupling_flag == 1 || coupling_flag == 2 {
        for i in (0..nstates) {
            for j in (0..nstates) {
                nonadibatic_slope[[i, j]] = (nonadiabatic_scalar[[i, j]]
                    - old_nonadiabatic_scalar[[i, j]])
                    / (n_delta as f64 * delta_rk);
            }
        }
    }

    // Get the electric field of the pulse
    let mut electric_field: Array1<f64> = Array1::zeros(4 * n_delta);
    if coupling_flag == 0 || coupling_flag == 2 {
        electric_field = get_electric_field(field_flag, 4 * n_delta, delta_rk, actual_step);
    }
    // println!("electric field {}",electric_field.slice(s![0..10]));

    // Integration
    let t_start: f64 = delta_rk * actual_step;
    let mut t_i: f64 = 0.0;
    let mut old_coefficients: Array1<c64> = coefficients.to_owned();

    for i in (0..n_delta) {
        t_i = i as f64 * delta_rk;
        let t_abs: f64 = t_start + t_i;

        // do runge kutta
        let new_coefficients: Array1<c64> = runge_kutta_integration(
            i + 1,
            t_i,
            old_coefficients.view(),
            energy,
            actual_step,
            nstates,
            actual_state,
            n_delta,
            delta_rk,
            dipole,
            old_nonadiabatic_scalar.view(),
            nonadibatic_slope.view(),
            coupling_flag,
            electric_field.view(),
        );
        old_coefficients = new_coefficients;
    }
    //let mut c_new: Array1<c64> = Array1::zeros(coefficients.raw_dim());
    let time: f64 = delta_rk * n_delta as f64;
    // for state in (0..nstates) {
    //     // c_new[state] =
    //     //     old_coefficients[state] * (-c64::new(0.0, 1.0) * energy[state] * t_i).exp();
    //     c_new[state] =
    //          old_coefficients[state] * (-c64::new(0.0, 1.0) * energy[state] * time).exp();
    // }
    let energy_compl: Array1<c64> = energy.mapv(|val| (-c64::new(0.0, 1.0) * val * time).exp());
    let c_new: Array1<c64> = old_coefficients * energy_compl;

    return c_new;
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

fn get_electric_field(field_flag: u8, nstep: usize, tstep: f64, nactstep: f64) -> Array1<f64> {
    let mut electric_field: Array1<f64> = Array1::zeros(nstep);
    // if field_flag == 0{
    //     // read electric field numerically
    // }
    if field_flag == 1 {
        // take parameters for gaussian pulses and
        // calculate the field analytically
        electric_field = get_analytic_field(nstep, tstep, nactstep);
    } else if field_flag == 3 {
        // reconstruct the field from spectral phases and amplitudes
    }
    return electric_field;
}

fn get_analytic_field(nstep: usize, tstep: f64, nactstep: f64) -> Array1<f64> {
    let mut electric_field: Array1<f64> = Array1::zeros(nstep);
    // read pulse parameters from external file
    let config_file_path: &Path = Path::new(PULSE_CONFIG);
    let mut config_string: String = if config_file_path.exists() {
        fs::read_to_string(config_file_path).expect("Unable to read config file")
    } else {
        String::from("")
    };
    // load the configuration
    let config: Pulse_Configuration = toml::from_str(&config_string).unwrap();
    if config_file_path.exists() == false {
        config_string = toml::to_string(&config).unwrap();
        fs::write(config_file_path, config_string).expect("Unable to write config file");
    }
    let param_1: f64 = config.pulse_param_1;
    let param_2: f64 = config.pulse_param_2;
    let param_3: f64 = config.pulse_param_3;
    let param_4: f64 = config.pulse_param_4;

    for step in (0..nstep) {
        let time: f64 = tstep * (nactstep + 0.25 * step as f64);
        electric_field[step] += param_1
            * (param_2 * (time - param_4)).cos()
            * (-param_3 * (time - param_4).powi(2)).exp();
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

#[test]
fn test_hopping_fortran() {
    let n_delta: usize = 2500;
    let delta_rk: f64 = 0.001653654933200;
    let nstates: usize = 3;
    let actual_state: usize = 2;
    let actual_step: f64 = 0.0;
    let energy: Array1<f64> = array![
        -76.382254427500001,
        -76.071502285500003,
        -75.969794950500003
    ];
    let coefficients: Array1<c64> =
        array![c64::new(0.0, 0.0), c64::new(0.0, 0.0), c64::new(1.0, 0.0)];
    let dipole: Array3<f64> = Array3::zeros((nstates, nstates, 3));
    let old_nonadiabatic_scalar: Array2<f64> = array![
        [0.0, 0.0, 0.00006639977399999999],
        [0.0, 0.0, 0.0],
        [-0.000066399773999999996, 0.0, 0.0]
    ];
    let nonadiabatic_scalar: Array2<f64> = array![
        [0.0, 0.0, 0.00006639977399999999],
        [0.0, 0.0, 0.0],
        [-0.000066399773999999996, 0.0, 0.0]
    ];
    let coupling_flag: i8 = 1;
    let field_flag: u8 = 1;

    let tmp = get_hopping_fortran(
        coefficients.view(),
        energy.view(),
        actual_step,
        nstates,
        actual_state,
        n_delta,
        delta_rk,
        dipole.view(),
        old_nonadiabatic_scalar.view(),
        nonadiabatic_scalar.view(),
        coupling_flag,
        field_flag,
    );
    println!("new coefficients {}", tmp);

    assert!(1 == 2);
}

#[test]
fn test_linalg() {
    use ndarray::Data;
    use ndarray_linalg::types::Lapack;
    use ndarray_linalg::*;
    let test: Array2<c64> = array![
        [c64::new(1.0, 0.0), c64::new(1.0, 0.0), c64::new(1.0, 0.0)],
        [c64::new(1.0, 0.0), c64::new(1.0, 0.0), c64::new(1.0, 0.0)],
        [c64::new(1.0, 0.0), c64::new(1.0, 0.0), c64::new(1.0, 0.0)],
    ];

    let test_2: Array2<f64> = Array::eye(3);
    let temp_2 = test_2.eig().unwrap();
    print_ifitworks(test_2);
    print_ifitworks(test);
    // let temp = test.eig().unwrap();

    fn print_ifitworks<S, A>(data: ArrayBase<S, Ix2>)
    where
        A: Scalar + Lapack,
        S: Data<Elem = A>,
    {
        println!("HALLLLOOO");
    }
}

#[test]
fn test_hopping_fortran_ethene() {
    let n_delta: usize = 2500;
    let delta_rk: f64 = 0.001653654933200;
    let nstates: usize = 3;
    let actual_state: usize = 0;
    let actual_step: f64 = 0.0;
    let energy: Array1<f64> = array![-78.4788953294, -78.4062414623, -78.4043408796];
    let coefficients: Array1<c64> =
        array![c64::new(1.0, 0.0), c64::new(0.0, 0.0), c64::new(0.0, 0.0)];
    let dipole: Array3<f64> = array![
        [
            [0.0, 0.0, 0.0],
            [0.000052, 0.003915, 0.004253],
            [0.016441, -0.002929, -0.003919]
        ],
        [
            [0.000052, 0.003915, 0.004253],
            [0.0, 0.0, 0.0],
            [1.158214, -0.015341, 0.000957]
        ],
        [
            [0.016441, -0.002929, -0.003919],
            [1.158214, -0.015341, 0.000957],
            [0.0, 0.0, 0.0]
        ]
    ];

    // let old_nonadiabatic_scalar: Array2<f64> = array![[0.0, 0.0003120921020448205, -0.000015860729434029015],
    //                                                  [-0.0003120921020448205, 0.0, 0.030229187960076172],
    //                                                  [0.000015860729434029015, -0.030229187960076172, 0.0]];
    let old_nonadiabatic_scalar: Array2<f64> = array![
        [0.0, 0.000312088821, -0.000016242621],
        [-0.000312088821, 0.0, 0.030222867039],
        [0.000016242621, -0.030222867039, 0.0]
    ];
    let nonadiabatic_scalar: Array2<f64> = array![
        [0.0, 0.000312088821, -0.000016242621],
        [-0.000312088821, 0.0, 0.030222867039],
        [0.000016242621, -0.030222867039, 0.0]
    ];
    let coupling_flag: i8 = 2;
    let field_flag: u8 = 1;

    let tmp = get_hopping_fortran(
        coefficients.view(),
        energy.view(),
        actual_step,
        nstates,
        actual_state,
        n_delta,
        delta_rk,
        dipole.view(),
        old_nonadiabatic_scalar.view(),
        nonadiabatic_scalar.view(),
        coupling_flag,
        field_flag,
    );
    println!("new coefficients {}", tmp);

    assert!(1 == 2);
}
