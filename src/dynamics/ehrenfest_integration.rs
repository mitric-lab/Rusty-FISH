use crate::initialization::Simulation;
use ndarray::prelude::*;
use ndarray_linalg::{c64, Eig, Inverse};

impl Simulation {
    pub fn ehrenfest_matrix_exponential(&self, exciton_couplings: ArrayView2<c64>) -> Array1<c64> {
        let mat: Array2<c64> =
            exciton_couplings.mapv(|val| -c64::new(0.0, 1.0) * val * self.stepsize);
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
            let new_coefficients = self.sod_step(
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

    fn initialize_sod(&self, dt: f64, exciton_couplings: ArrayView2<c64>) -> Array1<c64> {
        &self.coefficients + c64::new(0.0, 1.0) * dt * self.coefficients.dot(&exciton_couplings)
    }

    pub fn ehrenfest_rk_integration(&self, excitonic_couplings: ArrayView2<c64>) -> Array1<c64> {
        // set the stepsize of the RK-integration
        let n_delta: usize = self.config.hopping_config.integration_steps;
        // let delta_rk: f64 = self.stepsize / n_delta as f64;
        let delta_rk: f64 = 1.0 / n_delta as f64;

        // start the Runge-Kutta integration
        let mut old_coefficients: Array1<c64> = self.coefficients.clone();
        for _i in 0..n_delta {
            // do one step of the integration
            let new_coefficients: Array1<c64> = self.runge_kutta_ehrenfest(
                old_coefficients.view(),
                delta_rk,
                excitonic_couplings.view(),
            );
            old_coefficients = new_coefficients;
        }
        old_coefficients
    }

    /// Calculate one step of the 4th order Runge-Kutta method
    pub fn runge_kutta_ehrenfest(
        &self,
        coefficients: ArrayView1<c64>,
        delta_rk: f64,
        excitonic_couplings: ArrayView2<c64>,
    ) -> Array1<c64> {
        let mut k_1: Array1<c64> = self.rk_ehrenfest_helper(coefficients, excitonic_couplings);
        k_1 = k_1 * delta_rk;
        let tmp: Array1<c64> = &coefficients + &(&k_1 * 0.5);

        let mut k_2: Array1<c64> = self.rk_ehrenfest_helper(tmp.view(), excitonic_couplings);
        k_2 = k_2 * delta_rk;
        let tmp: Array1<c64> = &coefficients + &(&k_2 * 0.5);

        let mut k_3: Array1<c64> = self.rk_ehrenfest_helper(tmp.view(), excitonic_couplings);
        k_3 = k_3 * delta_rk;
        let tmp: Array1<c64> = &coefficients + &(&k_3 * 0.5);

        let mut k_4: Array1<c64> = self.rk_ehrenfest_helper(tmp.view(), excitonic_couplings);
        k_4 = k_4 * delta_rk;

        let new_coefficients: Array1<c64> =
            &coefficients + &((k_1 + k_2 * 2.0 + k_3 * 2.0 + k_4) * 1.0 / 6.0);
        new_coefficients
    }

    /// Calculate a coeffiecient k of the runge kutta method
    fn rk_ehrenfest_helper(
        &self,
        coefficients: ArrayView1<c64>,
        excitonic_couplings: ArrayView2<c64>,
    ) -> Array1<c64> {
        let f_new: Array1<c64> = coefficients.dot(&excitonic_couplings);

        f_new
    }
}
