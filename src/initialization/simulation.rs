use crate::constants;
use crate::initialization::system::SystemData;
use crate::initialization::velocities::*;
use crate::initialization::DynamicConfiguration;
use crate::interface::QuantumChemistryInterface;
use ndarray::prelude::*;
use ndarray_linalg::c64;

pub struct Simulation {
    pub stepsize: f64,
    pub actual_step: f64,
    pub actual_time: f64,
    pub total_mass: f64,
    pub config: DynamicConfiguration,
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
    pub saved_p_rand: Array2<f64>,
    pub saved_efactor: Array1<f64>,
    pub t_tot_last: Option<Array2<f64>>,
}

impl Simulation {
    pub fn new(system: &SystemData) -> Simulation {
        let config: DynamicConfiguration = system.config.clone();
        let stepsize_au: f64 = config.stepsize * constants::FS_TO_AU;

        // initialize coefficients
        let mut coefficients: Array1<c64> = Array1::zeros(config.nstates);
        coefficients[config.initial_state] = c64::from(1.0);
        // calculate total mass of the system
        let total_mass: f64 = system.masses.sum();

        // initiate parameters
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
        }
        println!("velocities {}", velocities);
        // for i in 0..system.n_atoms{
        //     println!("{},",velocities.slice(s![i,..]));
        // }

        Simulation {
            state: config.initial_state,
            actual_step: 0.0,
            actual_time: 0.0,
            stepsize: stepsize_au,
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
            saved_efactor: efactor,
            saved_p_rand: saved_p_rand,
            t_tot_last: None,
        }
    }
}
