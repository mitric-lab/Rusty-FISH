use ndarray::prelude::*;

use crate::constants;

pub trait Thermostat {
    fn scale_velocities(&mut self, vel: ArrayView2<f64>, kinetic_energy: f64) -> Array2<f64>;
}

pub struct NullThermostat {
    pub scaling: f64,
}

impl NullThermostat {
    pub fn default() -> Self {
        NullThermostat { scaling: 1.0 }
    }
}

impl Thermostat for NullThermostat {
    fn scale_velocities(&mut self, vel: ArrayView2<f64>, _kinetic_energy: f64) -> Array2<f64> {
        self.scaling * &vel
    }
}

pub struct BerendsenThermostat {
    pub tau: f64,
    pub dt: f64,
    pub n_atoms: f64,
    pub target_temperature: f64,
}

impl BerendsenThermostat {
    pub fn new(tau: f64, dt: f64, n_atoms: usize, temperature: f64) -> Self {
        let n_atoms: f64 = n_atoms as f64;
        BerendsenThermostat {
            tau,
            dt,
            n_atoms,
            target_temperature: temperature,
        }
    }

    fn get_temperature(&self, kinetic_energy: f64) -> f64 {
        let temperature: f64 = (23209.0 / (self.n_atoms * 3.0 - 6.0)) * kinetic_energy * 27.2114;
        temperature
    }
}

impl Thermostat for BerendsenThermostat {
    fn scale_velocities(&mut self, vel: ArrayView2<f64>, kinetic_energy: f64) -> Array2<f64> {
        let current_temperature: f64 = self.get_temperature(kinetic_energy);

        let scaling_factor: f64 = (1.0
            + (self.dt / self.tau) * (self.target_temperature / current_temperature - 1.0))
            .sqrt();

        scaling_factor * &vel
    }
}

pub struct NoseHoverThermostat {
    pub dt: f64,
    pub tau: f64,
    pub temperature: f64,
    pub n_atoms: usize,
    pub order: usize,
    pub chain_positions: Array1<f64>,
    pub chain_velocities: Array1<f64>,
    pub chain_accelerations: Array1<f64>,
    pub chain_particles: usize,
    pub weight_coefficients: Array1<f64>,
    pub integrator_steps: usize,
}

impl NoseHoverThermostat {
    pub fn new(
        tau: f64,
        dt: f64,
        n_atoms: usize,
        chain_particles: usize,
        integrator_steps: usize,
        temperature: f64,
    ) -> Self {
        let weight_coefficients: Array1<f64> = match chain_particles {
            3 => {
                let mut arr: Array1<f64> = Array1::zeros(3);
                arr[0] = 1.0 / (2.0 - 2.0_f64.powf(1.0 / 3.0));
                arr[1] = 1.0 - 2.0 * arr[0];
                arr[2] = arr[0];

                arr
            }
            5 => {
                let mut arr: Array1<f64> = Array1::zeros(5);
                let val = 1.0 / (4.0 - 4.0_f64.powf(1.0 / 3.0));
                arr[0] = val;
                arr[2] = 1.0 - 4.0 * val;
                arr[1] = val;
                arr[3] = val;
                arr[4] = val;

                arr
            }
            _ => panic!("Only 3rd and 5th order of NoseHover are available!"),
        };

        let chain_positions = Array::ones(chain_particles);
        let chain_velocities = Array::zeros(chain_particles);
        let chain_accelerations = Array::zeros(chain_particles);

        NoseHoverThermostat {
            dt,
            tau,
            n_atoms,
            order: chain_particles,
            temperature,
            chain_particles,
            weight_coefficients,
            chain_velocities,
            chain_accelerations,
            chain_positions,
            integrator_steps,
        }
    }

    fn get_scaling_factor(&mut self, kinetic_energy: f64) -> f64 {
        // factors k_bT and N_f k_b T
        let kt: f64 = constants::K_BOLTZMANN * self.temperature;
        let nkt: f64 = 3.0 * (self.n_atoms as f64 - 6.0) * kt;
        let len: usize = self.chain_particles - 1;

        let mut qmass: Array1<f64> = Array1::zeros(self.chain_particles);
        qmass[0] = nkt / (self.tau.powi(2));
        let val: f64 = kt / (self.tau.powi(2));

        for iter in 1..self.chain_particles {
            qmass[iter] = val;
        }

        let wdti: Array1<f64> = &self.weight_coefficients * self.dt;
        let wdti2: Array1<f64> = &wdti / 2.0;
        let wdti4: Array1<f64> = &wdti / 4.0;
        let wdti8: Array1<f64> = &wdti / 8.0;

        let mut scaling: f64 = 1.0;

        let ekin_2: f64 = kinetic_energy * 2.0;
        self.chain_accelerations[0] = (ekin_2 - nkt) / qmass[0];

        for _i in 0..self.integrator_steps {
            for j in 0..self.order {
                // update chain_velocities
                self.chain_velocities[len] += self.chain_accelerations[len] * wdti4[j];

                for k in 0..len - 1 {
                    let val: f64 = (-wdti8[j] * self.chain_velocities[len - k]).exp();
                    self.chain_velocities[len - k] = self.chain_velocities[len - k] * val.powi(2)
                        + wdti4[j] * self.chain_accelerations[len - k] * val;
                }

                let val: f64 = (-wdti2[j] * self.chain_velocities[0]).exp();
                scaling *= val;

                // update forces
                self.chain_accelerations[0] = (scaling.powi(2) * ekin_2 - nkt) / qmass[0];
                // update chain_positions
                self.chain_positions = &self.chain_positions + &self.chain_velocities * wdti2[j];

                // update thermostat velocities
                for k in 0..len - 1 {
                    let val: f64 = (-wdti8[j] * self.chain_velocities[k + 1]).exp();

                    self.chain_velocities[k] = self.chain_velocities[k] * val.powi(2)
                        + wdti4[j] * self.chain_accelerations[k] * val;
                    self.chain_accelerations[k + 1] =
                        (qmass[k] * self.chain_velocities[k].powi(2) - kt) / qmass[k + 1];
                }

                self.chain_velocities[len] += self.chain_accelerations[len] * wdti4[j];
            }
        }

        scaling
    }
}

impl Thermostat for NoseHoverThermostat {
    fn scale_velocities(&mut self, vel: ArrayView2<f64>, kinetic_energy: f64) -> Array2<f64> {
        let scale: f64 = self.get_scaling_factor(kinetic_energy);
        scale * &vel
    }
}
