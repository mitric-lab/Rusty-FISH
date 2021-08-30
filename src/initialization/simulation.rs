use crate::constants;
use crate::initialization::system::SystemData;
use crate::initialization::velocities::*;
use crate::initialization::Configuration;
use crate::interface::bagel::*;
use ndarray::prelude::*;
use ndarray_linalg::c64;

pub struct Simulation {
    pub stepsize: f64,
    pub delta_runge_kutta: f64,
    pub total_mass: f64,
    pub config: Configuration,
    pub coefficients: Array1<c64>,
    pub coordinates: Array2<f64>,
    pub masses: Array1<f64>,
    pub velocities: Array2<f64>,
    pub kinetic_energy: f64,
    pub n_atoms: usize,
    pub atomic_numbers: Vec<u8>,
    pub last_forces: Array3<f64>,
    pub friction: Array1<f64>,
    pub forces: Array2<f64>,
    pub energies: Array1<f64>,
    pub nonadiabatic_arr: Array3<f64>,
    pub nonadiabatic_arr_old: Array3<f64>,
    pub nonadiabatic_scalar: Array2<f64>,
    pub nonadiabatic_scalar_old: Array2<f64>,
    pub s_mat: Array2<f64>,
    pub dipole: Array3<f64>,
    pub dipole_old: Array3<f64>,
    pub state: usize,
    pub time_coupling: f64,
    pub handler: Option<Bagel_Handler>,
    pub saved_p_rand: Array2<f64>,
    pub saved_efactor: Array1<f64>,
    pub start_econst: f64,
}

impl Simulation {
    pub fn new(config: Configuration, system: &SystemData) -> Simulation {
        let stepsize_au: f64 = config.stepsize * constants::FS_TO_AU;
        let delta_runge_kutta: f64 = stepsize_au / config.n_small_steps as f64;

        // initialize coefficients
        let mut coefficients: Array1<c64> = Array1::zeros(config.nstates);
        coefficients[config.initial_state] = c64::from(1.0);
        // calculate total mass of the system
        let total_mass: f64 = system.masses.sum();

        // initiate parameters
        let econst: f64 = config.start_econst * constants::FS_TO_AU;
        let last_forces: Array3<f64> = Array3::zeros((3, system.n_atoms, 3));
        let forces: Array2<f64> = Array2::zeros((system.n_atoms, 3));
        let energies: Array1<f64> = Array1::zeros(config.nstates);
        let nonad_scalar: Array2<f64> = Array2::zeros((config.nstates, config.nstates));
        let s_mat: Array2<f64> = Array2::zeros((config.nstates, config.nstates));
        let dipole: Array3<f64> = Array3::zeros((config.nstates, config.nstates, 3));
        let nonad_arr: Array3<f64> = Array3::zeros((config.nstates, system.n_atoms, 3));
        let efactor: Array1<f64> = Array1::zeros(system.n_atoms);
        let saved_p_rand: Array2<f64> = Array2::zeros((system.n_atoms, 3));

        // set friction
        let mut friction: Array1<f64> = Array1::ones(system.n_atoms);
        friction = friction * config.friction;

        let mut velocities: Array2<f64> = Array2::zeros(system.coordinates.raw_dim());
        if config.velocity_generation == 0 {
            // initialize velocities from boltzmann distribution
            velocities = initialize_velocities(system, config.temperature);
        } else {
            // get velocities from inputs
        }

        Simulation {
            state: config.initial_state,
            stepsize: stepsize_au,
            delta_runge_kutta: delta_runge_kutta,
            total_mass: total_mass,
            time_coupling: config.time_coupling * constants::FS_TO_AU,
            config: config,
            coefficients: coefficients,
            coordinates: system.coordinates.clone(),
            masses: system.masses.clone(),
            velocities: velocities,
            kinetic_energy: 0.0,
            n_atoms: system.n_atoms,
            atomic_numbers: system.atomic_numbers.clone(),
            last_forces: last_forces,
            friction: friction,
            forces: forces,
            energies: energies,
            nonadiabatic_arr: nonad_arr.clone(),
            nonadiabatic_arr_old: nonad_arr,
            nonadiabatic_scalar: nonad_scalar.clone(),
            nonadiabatic_scalar_old: nonad_scalar,
            s_mat: s_mat,
            dipole: dipole.clone(),
            dipole_old: dipole,
            handler: None,
            saved_efactor: efactor,
            saved_p_rand: saved_p_rand,
            start_econst: econst,
        }
    }
}
