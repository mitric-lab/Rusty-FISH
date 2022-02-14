use crate::initialization::Simulation;
use ndarray::prelude::*;
use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::c64;
use rand::distributions::Standard;
use rand::prelude::*;
use std::ops::DivAssign;

impl Simulation {
    pub fn get_new_state(&mut self, old_coefficients: ArrayView1<c64>) {
        let nstates: usize = self.config.nstates;
        let mut occupations: Array1<f64> = Array1::zeros(nstates);
        let mut derivatives: Array1<f64> = Array1::zeros(nstates);

        for state in 0..nstates {
            occupations[state] =
                self.coefficients[state].re.powi(2) + self.coefficients[state].im.powi(2);
            derivatives[state] = (occupations[state]
                - (old_coefficients[state].re.powi(2) + old_coefficients[state].im.powi(2)))
                / self.stepsize;
        }
        let mut new_state: usize = self.state;
        if derivatives[self.state] < 0.0 {
            let mut hopping_probabilities: Array1<f64> = Array1::zeros(nstates);
            let mut probability: f64 = 0.0;
            let mut states_to_hopp: Vec<usize> = Vec::new();

            for k in 0..nstates {
                if derivatives[k] > 0.0 {
                    states_to_hopp.push(k);
                    probability += derivatives[k];
                }
            }
            for state in states_to_hopp {
                let tmp: f64 = old_coefficients[self.state].re.powi(2)
                    + old_coefficients[self.state].im.powi(2);
                hopping_probabilities[state] =
                    -1.0 * (derivatives[self.state] / tmp) * derivatives[state] * self.stepsize
                        / probability;
            }
            assert!(
                hopping_probabilities.sum() <= 1.0,
                "Total hopping probability bigger than 1.0!"
            );

            let random_number: f64 = StdRng::from_entropy().sample(Standard);

            let mut sum: f64 = 0.0;
            for state in 0..nstates {
                let prob: f64 = hopping_probabilities[state];
                if prob > 0.0 {
                    sum += prob;
                    if random_number < sum {
                        new_state = state;
                        // print coefficients here if required
                        break;
                    }
                }
            }
        }
        //  If the energy gap between the first excited state and the ground state
        //  approaches zero, because the trajectory has hit a conical intersection to
        //  the ground state, TD-DFT will break down. In this case, a transition
        //  to the ground state is forced.
        let threshold: f64 = 0.1 / 27.211;
        if new_state > 0 && self.config.hopping_config.force_switch_to_gs {
            let gap: f64 = self.energies[new_state] - self.energies[0];
            if gap < threshold {
                println!("Conical intersection to ground state reached.");
                println!("The trajectory will continue on the ground state.");
                new_state = 0;
                // if a conical intersection to the ground state is encountered
                // force the dynamic to stay in the ground state
                self.config.gs_dynamic = true;
            }
        }
        self.state = new_state;
    }

    pub fn get_decoherence_correction(&self, decoherence_constant: f64) -> Array1<c64> {
        // decoherence correction according to eqn. (17) in
        // G. Granucci, M. Persico,
        // "Critical appraisal of the fewest switches algorithm for surface hopping",
        // J. Chem. Phys. 126, 134114 (2007)
        // If the trajectory is in the current state K, the coefficients of the other
        // states J != K are made to decay exponentially, C'_J = exp(-dt/tau_JK) C_J.
        // The decay time is proportional to the inverse of the energy gap |E_J-E_K|,
        // so that the coherences C_J*C_K decay very quickly if the energy gap between
        // the two states is large. The electronic transitions become irreversible.
        // CAUTION! THIS SHOULD NOT BE USED DURING THE TIME A FIELD IS SWITCHED ON!

        let mut sm: f64 = 0.0;
        let mut new_coefficients: Array1<c64> = self.coefficients.clone();
        for state in 0..self.config.nstates {
            if state == self.state {
                let tauij: f64 = 1.0 / (self.energies[state] - self.energies[self.state]).abs()
                    * (1.0 + decoherence_constant / self.kinetic_energy);
                new_coefficients[state] *= (-self.stepsize / tauij).exp();
                sm += new_coefficients[state].re.powi(2) + new_coefficients[state].im.powi(2);
            }
        }
        let tmp: f64 =
            new_coefficients[self.state].re.powi(2) + new_coefficients[self.state].im.powi(2);
        new_coefficients[self.state] = new_coefficients[self.state] * (1.0 - sm).sqrt() / tmp;

        new_coefficients
    }

    // Rescaling routines for combined nonadiabatic/field-coupled dynamics
    // self.scale is needed to decide how large a part of the velocity is rescaled
    // for pure nonadiabatic dynamics it can be set to 1.0
    pub fn uniformly_rescaled_velocities(&self, old_state: usize) -> (Array2<f64>, usize) {
        // hop is rejected when kinetic energy is too low
        let mut state: usize = self.state;
        let mut new_velocities: Array2<f64> = self.velocities.clone();
        if self.state > old_state
            && (self.energies[self.state] - self.energies[old_state]) > self.kinetic_energy
        {
            state = old_state;
        } else if self.kinetic_energy > 0.0 {
            let vel_scale: f64 = ((self.kinetic_energy
                + (self.energies[old_state] - self.energies[self.state]))
                / self.kinetic_energy)
                .sqrt();
            new_velocities *= vel_scale;
        }
        (new_velocities, state)
    }

    pub fn rescaled_velocities(&self, old_state: usize, last_state: usize) -> (Array2<f64>, usize) {
        let mut factor: f64 = 1.0;
        // the following is important to find the right coupling vector!
        // if the hop occurs to lower states
        let new_state: usize = self.state;
        let first_state: usize = 0;
        let nonadiabatic_new: ArrayView3<f64> = self.nonadiabatic_arr.view();

        let mut jj: usize = new_state;
        let mut ii: usize = old_state;
        if new_state < old_state {
            jj = old_state;
            ii = new_state;
            factor = -1.0;
        }
        let mut state: usize = new_state;

        let delta_e: f64 = self.energies[old_state] - self.energies[new_state];
        let mut k: usize = 0;
        let mut flag: usize = 0;

        for j in first_state + 1..last_state + 1 {
            for i in first_state..j {
                if j == jj && i == ii {
                    flag = 1;
                    break;
                }
                k += 1;
            }
            if flag == 1 {
                break;
            }
        }
        let mut new_velocities: Array2<f64> = self.velocities.clone();
        if k < nonadiabatic_new.dim().0 {
            // rescaling only if coupling between initial and final states exists
            let mut mass_weigh_nad: Array2<f64> = factor * &nonadiabatic_new.slice(s![k, .., ..]);

            for i in 0..self.n_atoms {
                mass_weigh_nad
                    .slice_mut(s![i, ..])
                    .div_assign(self.masses[i]);
            }

            let mut a: f64 = 0.0;
            for i in 0..self.n_atoms {
                a += nonadiabatic_new
                    .slice(s![k, i, ..])
                    .dot(&nonadiabatic_new.slice(s![k, i, ..]))
                    / self.masses[i];
            }
            a *= 0.5;

            let mut b: f64 = 0.0;
            for i in 0..self.n_atoms {
                b += self
                    .velocities
                    .slice(s![i, ..])
                    .dot(&nonadiabatic_new.slice(s![k, i, ..]));
            }

            let val: f64 = b.powi(2) + 4.0 * a * delta_e;

            let gamma: f64 = if val < 0.0 {
                state = old_state;
                b / a
            } else if b < 0.0 {
                (b + val.sqrt()) / (2.0 * a)
            } else {
                (b - val.sqrt()) / (2.0 * a)
            };
            new_velocities = &self.velocities - &(gamma * mass_weigh_nad);
        }
        (new_velocities, state)
    }

    pub fn scale_velocities_temperature(&self) -> Array2<f64> {
        let curr_temperature: f64 =
            (23209.0 / (self.n_atoms as f64 * 3.0 - 6.0)) * self.kinetic_energy * 27.2114;
        let scaling_factor: f64 = (1.0
            + (self.stepsize / self.time_coupling)
                * (self.config.temperature / curr_temperature - 1.0))
            .sqrt();
        let new_velocities: Array2<f64> = scaling_factor * &self.velocities;
        new_velocities
    }

    pub fn scale_velocities_const_energy(
        &self,
        old_state: usize,
        old_kinetic_energy: f64,
        old_potential_energy: f64,
    ) -> Array2<f64> {
        // The velocities are rescaled so that energy conservation
        // between two time-steps is fulfilled exactly.
        let mut new_velocities: Array2<f64> = Array2::zeros(self.velocities.raw_dim());
        if self.state != old_state {
            let scaling_factor: f64 = ((old_kinetic_energy
                + (old_potential_energy - self.energies[self.state]))
                / self.kinetic_energy)
                .sqrt();
            assert!(
                (scaling_factor - 1.0).abs() < 1.0e-1,
                "Total energy is not conserved!"
            );

            new_velocities = scaling_factor * &self.velocities;
        }
        new_velocities
    }
}

pub fn normalize_coefficients(coefficients: ArrayView1<c64>) -> Array1<c64> {
    let mut norm: f64 = 0.0;
    let nstates: usize = coefficients.len();

    for state in 0..nstates {
        norm += coefficients[state].re.powi(2) + coefficients[state].im.powi(2);
    }
    let new_coefficients: Array1<c64> = coefficients.to_owned() / norm.sqrt();
    new_coefficients
}
