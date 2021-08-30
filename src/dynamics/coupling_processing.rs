use crate::constants;
use ndarray::prelude::*;
use ndarray::{array, Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::{into_col, into_row, solve, Inverse};
use rand::Rng;
use rand_distr::{Distribution, Normal};

pub fn align_dipoles(dipoles: ArrayView3<f64>) -> Array3<f64> {
    let mut new_dipoles: Array3<f64> = dipoles.to_owned();
    for k in (0..dipoles.dim().0) {
        for l in (0..dipoles.dim().1) {
            let sign: f64 = dipoles
                .slice(s![k, l, ..])
                .dot(&dipoles.slice(s![k, l, ..]));
            if sign < 0.0 {
                for m in (0..3) {
                    new_dipoles[[k, l, m]] = -1.0 * dipoles[[k, l, m]];
                }
            }
        }
    }
    return new_dipoles;
}

pub fn align_nonadiabatic_coupling(
    nonadiabatic_old: ArrayView3<f64>,
    nonadiabatic_new: ArrayView3<f64>,
    n_at: usize,
) -> Array3<f64> {
    let k_0: usize = nonadiabatic_old.dim().0;
    let k_1: usize = nonadiabatic_new.dim().0;
    let min: usize = k_0.min(k_1);
    let mut nonad_new: Array3<f64> = nonadiabatic_new.to_owned();

    let shape_0: usize = nonadiabatic_new.dim().1;
    let shape_1: usize = nonadiabatic_new.dim().2;
    for k in (0..min) {
        let temp_nad_new: ArrayView1<f64> = nonadiabatic_new
            .slice(s![k, .., ..])
            .into_shape(shape_0 * shape_1)
            .unwrap();
        let temp_nad_old: ArrayView1<f64> = nonadiabatic_old
            .slice(s![k, .., ..])
            .into_shape(shape_0 * shape_1)
            .unwrap();
        let sign: f64 = temp_nad_old.dot(&temp_nad_new);

        if sign < 0.0 {
            for m in (0..n_at) {
                for mm in (0..3) {
                    nonad_new[[k, m, mm]] = -1.0 * nonadiabatic_new[[k, m, mm]];
                }
            }
        }
    }
    return nonad_new;
}

pub fn get_nonadiabatic_scalar_coupling(
    nstates: usize,
    first_state: usize,
    last_state: usize,
    nonadiabatic_new: ArrayView3<f64>,
    velocities: ArrayView2<f64>,
) -> Array2<f64> {
    let shape_0: usize = nonadiabatic_new.dim().1;
    let shape_1: usize = nonadiabatic_new.dim().2;

    let mut nonadibatic_scalar: Array2<f64> = Array2::zeros((nstates, nstates));
    let mut k: usize = 0;
    for state in (0..nstates) {
        for i in (0..state) {
            if (first_state + 1..last_state + 1).contains(&state)
                && (first_state..state).contains(&i)
            {
                let temp_nad: ArrayView1<f64> = nonadiabatic_new
                    .slice(s![k, .., ..])
                    .into_shape(shape_0 * shape_1)
                    .unwrap();

                let temp_vel: ArrayView1<f64> = velocities.into_shape(shape_0 * shape_1).unwrap();

                let temp_scalar: f64 = temp_nad.dot(&temp_vel);

                nonadibatic_scalar[[state, i]] = 1.0 * temp_scalar;
                nonadibatic_scalar[[i, state]] = -1.0 * nonadibatic_scalar[[state, i]];
                k = k + 1;
            }
        }
    }

    return nonadibatic_scalar;
}

pub fn get_gradient_norm(forces: ArrayView2<f64>) -> f64 {
    let n_at: usize = forces.dim().0;
    let tmp_force: ArrayView1<f64> = forces.into_shape(3 * n_at).unwrap();
    let grad_norm: f64 = tmp_force.dot(&tmp_force).abs();
    return grad_norm;
}
