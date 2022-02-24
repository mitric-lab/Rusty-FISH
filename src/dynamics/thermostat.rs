use ndarray::prelude::*;

pub trait Thermostat {
    fn scale_velocities(&self, vel: ArrayView2<f64>, kinetic_energy: f64) -> Array2<f64>;
    fn get_temperature(&self, kinetic_energy: f64) -> f64;
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
}

impl Thermostat for BerendsenThermostat {
    fn scale_velocities(&self, vel: ArrayView2<f64>, kinetic_energy: f64) -> Array2<f64> {
        let current_temperature: f64 = self.get_temperature(kinetic_energy);

        let scaling_factor: f64 = (1.0
            + (self.dt / self.tau) * (self.target_temperature / current_temperature - 1.0))
            .sqrt();

        scaling_factor * &vel
    }

    fn get_temperature(&self, kinetic_energy: f64) -> f64 {
        let temperature: f64 = (23209.0 / (self.n_atoms * 3.0 - 6.0)) * kinetic_energy * 27.2114;
        temperature
    }
}
