use crate::initialization::Simulation;
use ndarray::prelude::*;
use ndarray_linalg::{c64, Eig, Inverse};
use std::time::Instant;

impl Simulation {
    pub fn ehrenfest_matrix_exponential(&self, exciton_couplings: ArrayView2<c64>) -> Array1<c64> {
        let mat: Array2<c64> =
            exciton_couplings.mapv(|val| -c64::new(0.0, 1.0) * val * self.stepsize);
        let (eig, eig_vec): (Array1<c64>, Array2<c64>) = mat.eig().unwrap();
        let diag: Array1<c64> = eig.mapv(|val| val.exp());
        let mat: Array2<c64> = eig_vec.dot(&Array::from_diag(&diag).dot(&eig_vec.inv().unwrap()));

        mat.dot(&self.coefficients)
    }

    pub fn ehrenfest_matrix_exponential_nacme(
        &self,
        exciton_couplings: ArrayView2<c64>,
    ) -> Array1<c64> {
        let mut mat: Array2<c64> =
            &(c64::new(0.0, 1.0) * &exciton_couplings) - &self.nonadiabatic_scalar;
        mat = mat * (-1.0 * self.stepsize);
        let (eig, eig_vec): (Array1<c64>, Array2<c64>) = mat.eig().unwrap();
        let diag: Array1<c64> = eig.mapv(|val| val.exp());
        let mat: Array2<c64> = eig_vec.dot(&Array::from_diag(&diag).dot(&eig_vec.inv().unwrap()));

        mat.dot(&self.coefficients)
    }

    pub fn ehrenfest_sod_integration(&self, exciton_couplings: ArrayView2<c64>) -> Array1<c64> {
        // set the stepsize of the RK-integration
        let n_delta: usize = self.config.ehrenfest_config.integration_steps;
        let dt: f64 = self.stepsize / n_delta as f64;

        let mut coefficients: Array1<c64> = self.coefficients.clone();
        let mut prev: Array1<c64> = self.initialize_sod(dt, exciton_couplings.view());
        for _i in 0..n_delta {
            let new_coefficients = self.sod_step_nacme(
                dt,
                exciton_couplings.view(),
                coefficients.view(),
                prev.view(),
            );
            prev = coefficients;
            coefficients = new_coefficients;
        }
        coefficients
    }

    fn sod_step(
        &self,
        dt: f64,
        exciton_couplings: ArrayView2<c64>,
        coeff: ArrayView1<c64>,
        prev_coeff: ArrayView1<c64>,
    ) -> Array1<c64> {
        &prev_coeff - 2.0 * c64::new(0.0, 1.0) * dt * coeff.dot(&exciton_couplings)
    }

    fn sod_step_nacme(
        &self,
        dt: f64,
        exciton_couplings: ArrayView2<c64>,
        coeff: ArrayView1<c64>,
        prev_coeff: ArrayView1<c64>,
    ) -> Array1<c64> {
        &prev_coeff
            - (&(c64::new(0.0, 1.0) * &exciton_couplings) - &self.nonadiabatic_scalar).dot(&coeff)
                * 2.0
                * dt
    }

    fn initialize_sod(&self, dt: f64, exciton_couplings: ArrayView2<c64>) -> Array1<c64> {
        &self.coefficients + c64::new(0.0, 1.0) * dt * self.coefficients.dot(&exciton_couplings)
    }

    fn initialize_sod_nacme(&self, dt: f64, exciton_couplings: ArrayView2<c64>) -> Array1<c64> {
        &self.coefficients
            + (&(c64::new(0.0, 1.0) * &exciton_couplings) - &self.nonadiabatic_scalar)
                .dot(&self.coefficients)
                * dt
    }
    pub fn ehrenfest_rk_integration(&self, excitonic_couplings: ArrayView2<c64>) -> Array1<c64> {
        // set the stepsize of the RK-integration
        let n_delta: usize = self.config.ehrenfest_config.integration_steps;
        // let delta_rk: f64 = self.stepsize / n_delta as f64;
        let delta_rk: f64 = self.stepsize / n_delta as f64;

        // create energy meshgrid
        let nstates: usize = self.config.nstates;
        let energy_arr_tmp: Array2<f64> = self.energies.clone().insert_axis(Axis(1));
        let mesh_1: ArrayView2<f64> = energy_arr_tmp.broadcast((nstates, nstates)).unwrap();
        let energy_mesh: Array2<f64> = &mesh_1.clone() - &mesh_1.t();

        let excitonic_couplings: Array2<c64> = &(c64::new(0.0, 1.0)
            * &(&excitonic_couplings - Array::from_diag(&excitonic_couplings.diag())))
            + &self.nonadiabatic_scalar;

        let timer: Instant = Instant::now();
        // start the Runge-Kutta integration
        let mut old_coefficients: Array1<c64> = self.coefficients.clone();
        for i in 0..n_delta {
            // do one step of the integration
            let t_i: f64 = i as f64 * delta_rk;
            old_coefficients = self.runge_kutta_ehrenfest(
                old_coefficients.view(),
                delta_rk,
                t_i,
                excitonic_couplings.view(),
                energy_mesh.view(),
            );
        }
        println!("Time 1: {}", timer.elapsed().as_secs_f64());
        // calculate the new coefficients
        let time: f64 = delta_rk * n_delta as f64;
        let energy_compl: Array1<c64> = self
            .energies
            .mapv(|val| (-c64::new(0.0, 1.0) * val * time).exp());
        let c_new: Array1<c64> = old_coefficients * energy_compl;
        println!("Time 2: {}", timer.elapsed().as_secs_f64());

        c_new
    }

    /// Calculate one step of the 4th order Runge-Kutta method
    pub fn runge_kutta_ehrenfest(
        &self,
        coefficients: ArrayView1<c64>,
        delta_rk: f64,
        time: f64,
        excitonic_couplings: ArrayView2<c64>,
        energy_mesh: ArrayView2<f64>,
    ) -> Array1<c64> {
        let mut k_1: Array1<c64> =
            self.rk_ehrenfest_helper(coefficients, time, excitonic_couplings, energy_mesh);
        k_1 = k_1 * delta_rk;
        let tmp: Array1<c64> = &coefficients + &(&k_1 * 0.5);

        let mut k_2: Array1<c64> = self.rk_ehrenfest_helper(
            tmp.view(),
            time + 0.5 * delta_rk,
            excitonic_couplings,
            energy_mesh,
        );
        k_2 = k_2 * delta_rk;
        let tmp: Array1<c64> = &coefficients + &(&k_2 * 0.5);

        let mut k_3: Array1<c64> = self.rk_ehrenfest_helper(
            tmp.view(),
            time + 0.5 * delta_rk,
            excitonic_couplings,
            energy_mesh,
        );
        k_3 = k_3 * delta_rk;
        let tmp: Array1<c64> = &coefficients + &k_3;

        let mut k_4: Array1<c64> = self.rk_ehrenfest_helper(
            tmp.view(),
            time + delta_rk,
            excitonic_couplings,
            energy_mesh,
        );
        k_4 = k_4 * delta_rk;

        let new_coefficients: Array1<c64> =
            &coefficients + &((k_1 + k_2 * 2.0 + k_3 * 2.0 + k_4) * 1.0 / 6.0);
        new_coefficients
    }

    /// Calculate a coeffiecient k of the runge kutta method
    fn rk_ehrenfest_helper(
        &self,
        coefficients: ArrayView1<c64>,
        time: f64,
        exciton_couplings: ArrayView2<c64>,
        energy_difference: ArrayView2<f64>,
    ) -> Array1<c64> {
        let de: Array2<c64> = energy_difference.mapv(|val| (c64::new(0.0, 1.0) * val * time).exp());
        let h: Array2<c64> = -de * exciton_couplings;
        let f_new: Array1<c64> = h.dot(&coefficients);

        f_new
    }
}
