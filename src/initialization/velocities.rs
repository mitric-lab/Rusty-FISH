use crate::constants;
use crate::initialization::SystemData;
use ndarray::Array2;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Initialize the velocities from a Boltzmann distribution.
pub struct boltzmann_velocities {
    temperature: f64,
    dist: Normal<f64>,
}

impl boltzmann_velocities {
    pub fn new(temperature: f64) -> boltzmann_velocities {
        let dist = Normal::new(0.0, f64::sqrt(constants::K_BOLTZMANN * temperature))
            .expect("Error regarding the distribution!");
        boltzmann_velocities {
            temperature: temperature,
            dist: dist,
        }
    }
}

pub fn initialize_velocities(system: &SystemData, temperature: f64) -> Array2<f64> {
    let boltzmann: boltzmann_velocities = boltzmann_velocities::new(temperature);
    let mut velocities: Array2<f64> = Array2::zeros(system.coordinates.raw_dim());

    for atom in (0..system.n_atoms) {
        let mass_inv: f64 = 1.0 / system.masses[atom];
        velocities[[atom, 0]] =
            f64::sqrt(mass_inv) * boltzmann.dist.sample(&mut rand::thread_rng());
        velocities[[atom, 1]] =
            f64::sqrt(mass_inv) * boltzmann.dist.sample(&mut rand::thread_rng());
        velocities[[atom, 2]] =
            f64::sqrt(mass_inv) * boltzmann.dist.sample(&mut rand::thread_rng());
    }
    return velocities;
}
