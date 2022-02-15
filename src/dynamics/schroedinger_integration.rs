use crate::initialization::PulseConfiguration;
use crate::initialization::Simulation;
use ndarray::prelude::*;
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::{c64, Eig, Eigh, Inverse, UPLO};

impl Simulation {
    /// Obtain the new coefficients of the states by utilizing a Runge-Kutta 4th order scheme
    pub fn new_coefficients(&self) -> Array1<c64> {
        // get the nonadiabatic scalar couplings
        let old_nonadiabatic_scalar: Array2<f64> = -1.0 * &self.nonadiabatic_scalar_old;
        let nonadiabatic_scalar: Array2<f64> = -1.0 * &self.nonadiabatic_scalar;

        // set the stepsize of the RK-integration
        let n_delta: usize = self.config.hopping_config.integration_steps;
        let delta_rk: f64 = self.stepsize / n_delta as f64;
        let coupling_flag: i8 = self.config.hopping_config.coupling_flag;

        // calculate the nonadiabatic slope
        let mut nonadibatic_slope: Array2<f64> = Array2::zeros(nonadiabatic_scalar.raw_dim());
        if coupling_flag == 1 || coupling_flag == 2 {
            for i in 0..self.config.nstates {
                for j in 0..self.config.nstates {
                    nonadibatic_slope[[i, j]] = (nonadiabatic_scalar[[i, j]]
                        - old_nonadiabatic_scalar[[i, j]])
                        / (n_delta as f64 * delta_rk);
                }
            }
        }

        // Get the electric field of the laser pulse
        let mut electric_field: Array1<f64> = Array1::zeros(4 * n_delta);
        if coupling_flag == 0 || coupling_flag == 2 {
            electric_field = get_analytic_field(
                &self.config.pulse_config,
                4 * n_delta,
                delta_rk,
                self.actual_step,
            );
        }

        // start the Runge-Kutta integration
        let t_start: f64 = delta_rk * self.actual_step;
        let mut old_coefficients: Array1<c64> = self.coefficients.clone();
        for i in 0..n_delta {
            let t_i: f64 = i as f64 * delta_rk;
            let _t_abs: f64 = t_start + t_i;

            // do one step of the integration
            let new_coefficients: Array1<c64> = self.runge_kutta_integration(
                i + 1,
                t_i,
                old_coefficients.view(),
                delta_rk,
                nonadibatic_slope.view(),
                electric_field.view(),
            );
            old_coefficients = new_coefficients;
        }
        // calulate the new coefficients
        let time: f64 = delta_rk * n_delta as f64;
        let energy_compl: Array1<c64> = self
            .energies
            .mapv(|val| (-c64::new(0.0, 1.0) * val * time).exp());
        let c_new: Array1<c64> = old_coefficients * energy_compl;

        c_new
    }

    /// The coefficients of the electronic wavefunction are propagated
    /// in the local diabatic basis as explained in
    /// [1]  JCP 114, 10608 (2001) and
    /// [2]  JCP 137, 22A514 (2012)
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
        for ii in 0..h_interp.dim().0 {
            h_interp[[ii, ii]] -= h_00_val;
        }

        // propagator in diabatic basis, see eqn. (11) in [1]
        let u_1: Array2<c64> = h_interp.mapv(|val| -c64::new(0.0, 1.0) * val * self.stepsize);
        let (eig, eig_vec): (Array1<c64>, Array2<c64>) = u_1.eig().unwrap();
        let diag: Array1<c64> = eig.mapv(|val| val.exp());
        let u_mat: Array2<c64> = eig_vec.dot(&Array::from_diag(&diag).dot(&eig_vec.inv().unwrap()));

        // at the beginning of the time step the adiabatic and diabatic basis is assumed to coincide
        // new electronic coefficients c(t+dt) in the adiabatic basis
        let complex_t_inv: Array2<c64> = t_inv.mapv(|val| val * c64::new(1.0, 0.0));
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

        (c_1, h_diab, ttot_last)
    }

    /// Calculate one step of the 4th order Runge-Kutta method
    pub fn runge_kutta_integration(
        &self,
        iterator: usize,
        time: f64,
        coefficients: ArrayView1<c64>,
        delta_rk: f64,
        nonadiabatic_slope: ArrayView2<f64>,
        efield: ArrayView1<f64>,
    ) -> Array1<c64> {
        let n: usize = 4 * iterator - 3;
        let mut k_1: Array1<c64> =
            self.runge_kutta_helper(time, coefficients, n, nonadiabatic_slope, efield);
        k_1 = k_1 * delta_rk;
        let tmp: Array1<c64> = &coefficients + &(&k_1 * 0.5);

        let n: usize = 4 * iterator - 2;
        let mut k_2: Array1<c64> = self.runge_kutta_helper(
            time + 0.5 * delta_rk,
            tmp.view(),
            n,
            nonadiabatic_slope,
            efield,
        );
        k_2 = k_2 * delta_rk;
        let tmp: Array1<c64> = &coefficients + &(&k_2 * 0.5);

        let n: usize = 4 * iterator - 1;
        let mut k_3: Array1<c64> = self.runge_kutta_helper(
            time + 0.5 * delta_rk,
            tmp.view(),
            n,
            nonadiabatic_slope,
            efield,
        );
        k_3 = k_3 * delta_rk;
        let tmp: Array1<c64> = &coefficients + &(&k_3 * 0.5);

        let n: usize = 4 * iterator;
        let mut k_4: Array1<c64> =
            self.runge_kutta_helper(time + delta_rk, tmp.view(), n, nonadiabatic_slope, efield);
        k_4 = k_4 * delta_rk;

        let new_coefficients: Array1<c64> =
            &coefficients + &((k_1 + k_2 * 2.0 + k_3 * 2.0 + k_4) * 1.0 / 6.0);
        new_coefficients
    }

    /// Calculate a coeffiecient k of the runge kutta method
    fn runge_kutta_helper(
        &self,
        time: f64,
        coefficients: ArrayView1<c64>,
        n: usize,
        nonadiabatic_slope: ArrayView2<f64>,
        efield: ArrayView1<f64>,
    ) -> Array1<c64> {
        let _f: Array1<c64> = Array1::zeros(coefficients.raw_dim());
        let mut non_adiabatic: Array2<f64> = Array2::zeros(nonadiabatic_slope.raw_dim());
        let mut field_coupling: Array2<f64> = Array2::zeros(nonadiabatic_slope.raw_dim());
        let nstates: usize = self.config.nstates;

        // coupling_flag =0: field coupling only; 1: nonadiabatic coupling only; 2: both
        if self.config.hopping_config.coupling_flag == 0
            || self.config.hopping_config.coupling_flag == 2
        {
            field_coupling = get_field_coupling(
                self.dipole.view(),
                n,
                self.config.nstates,
                efield,
                self.config.pulse_config.rotational_averaging,
            );
        }

        if self.config.hopping_config.coupling_flag == 1
            || self.config.hopping_config.coupling_flag == 2
        {
            let old_nonadiabatic_scalar: Array2<f64> = -1.0 * &self.nonadiabatic_scalar_old;
            non_adiabatic = get_nonadiabatic_coupling(
                time,
                nonadiabatic_slope,
                old_nonadiabatic_scalar.view(),
                nstates,
            );
        }
        // create energy difference array
        let energy_arr_tmp: Array2<f64> = self.energies.clone().insert_axis(Axis(1));
        let mesh_1: ArrayView2<f64> = energy_arr_tmp.broadcast((nstates, nstates)).unwrap();
        let energy_difference: Array2<f64> = &mesh_1.clone() - &mesh_1.t();

        // alternative way instead of iteration
        let de: Array2<c64> = energy_difference.mapv(|val| (c64::new(0.0, 1.0) * val * time).exp());
        let mut incr: Array2<f64> = Array2::zeros((nstates, nstates));
        incr = incr + non_adiabatic;
        let incr_complex = field_coupling.mapv(|val| c64::new(0.0, -1.0) * val) + incr;
        let h: Array2<c64> = de * incr_complex;
        let f_new: Array1<c64> = h.dot(&coefficients);

        f_new
    }
}

/// Obtain the nonadiabatic coupling for a integration step of the RK method
fn get_nonadiabatic_coupling(
    time: f64,
    nonadiabatic_slope: ArrayView2<f64>,
    old_nonadiabatic_scalar: ArrayView2<f64>,
    nstates: usize,
) -> Array2<f64> {
    let mut nonadiabatic: Array2<f64> = Array::zeros(old_nonadiabatic_scalar.raw_dim());
    for i in 0..nstates {
        for j in 0..nstates {
            nonadiabatic[[i, j]] =
                old_nonadiabatic_scalar[[i, j]] + nonadiabatic_slope[[i, j]] * time;
        }
    }
    nonadiabatic
}

/// Obtain the coupling of the electric field for a integration step of the RK method
fn get_field_coupling(
    dipole: ArrayView3<f64>,
    n: usize,
    nstates: usize,
    efield: ArrayView1<f64>,
    rot_avg: bool,
) -> Array2<f64> {
    let mut coupling: Array2<f64> = Array2::zeros((nstates, nstates));
    let efield_index: usize = n - 1;

    // use the rotationally averaged field coupling
    if rot_avg {
        for i in 0..nstates {
            for j in 0..nstates {
                let dipole_xyz: ArrayView1<f64> = dipole.slice(s![i, j, ..]);
                let dipole_norm: f64 = (dipole_xyz.dot(&dipole_xyz)).sqrt();
                coupling[[i, j]] = dipole_norm;
            }
        }
        coupling = coupling * (-1.0 / (3.0_f64.sqrt())) * efield[efield_index];
    }
    // use a fixed field polarization
    else {
        let evec: Array1<f64> = (-1.0 / 3.0_f64.sqrt()) * Array1::ones(3);
        let evec_normalized = &evec / (evec.dot(&evec)).sqrt();

        coupling = dipole
            .into_shape([nstates * nstates, 3])
            .unwrap()
            .dot(&evec_normalized)
            .into_shape([nstates, nstates])
            .unwrap();

        coupling *= efield[efield_index];
    }
    coupling
}

/// Obtain the electric field of the laser pulse given by the parameters of the [PulseConfiguration].
/// Calculate the laser pulse using the equation
/// E = E_0 * cos(ω (t-t0)) * exp(-α (t-t0)^2)
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

    for step in 0..nstep {
        let time: f64 = tstep * (nactstep + 0.25 * step as f64);
        electric_field[step] +=
            e0 * (omega * (time - t0)).cos() * (-alpha * (time - t0).powi(2)).exp();
    }
    electric_field
}
