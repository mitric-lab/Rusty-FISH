use crate::constants;
use crate::dynamics::thermostat::BerendsenThermostat;
use crate::dynamics::thermostat::NoseHooverThermostat;
use crate::dynamics::thermostat::NullThermostat;
use crate::dynamics::thermostat::Thermostat;
use crate::initialization::system::SystemData;
use crate::initialization::velocities::*;
use crate::initialization::DynamicConfiguration;
use ndarray::prelude::*;
use ndarray_linalg::c64;

/// Struct that holds the [DynamicConfiguration] and the other necessary
/// arguments, which are required for the molecular dynamics
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
    pub saved_p_rand: Array2<f64>,
    pub saved_efactor: Array1<f64>,
    pub t_tot_last: Option<Array2<f64>>,
    pub thermostat: Box<dyn Thermostat>,
}

impl Simulation {
    /// Initialize the struct [Simulation] from the [SystemData]
    /// Create all required arrays and initialize the velocities
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
        friction *= config.friction;

        let mut velocities: Array2<f64> = Array2::zeros(system.coordinates.raw_dim());
        if config.use_boltzmann_velocities {
            // initialize velocities from boltzmann distribution
            velocities = initialize_velocities(system, config.thermostat_config.temperature);
        } else {
            // read velocities from a file
            // TODO: read from file
        }

        let thermostat: Box<dyn Thermostat> = if !config.thermostat_config.use_thermostat {
            Box::new(NullThermostat::new(system.n_atoms))
        } else if config.thermostat_config.thermostat_type == *"Berendsen" {
            Box::new(BerendsenThermostat::new(
                config.thermostat_config.time_coupling * constants::FS_TO_AU,
                stepsize_au,
                system.n_atoms,
                config.thermostat_config.temperature,
            ))
        } else if config.thermostat_config.thermostat_type == *"NoseHoover" {
            Box::new(NoseHooverThermostat::new(
                config.thermostat_config.time_coupling,
                stepsize_au,
                system.n_atoms,
                config.thermostat_config.nh_chain_length,
                config.thermostat_config.nh_steps,
                config.thermostat_config.temperature,
            ))
        } else {
            panic!("Unknown thermostat! Choose between 'Berendsen' and 'NoseHoover'! Or set the parameter 'use_thermostat' to 'false'!")
        };

        Simulation {
            state: config.initial_state,
            actual_step: 0.0,
            actual_time: 0.0,
            stepsize: stepsize_au,
            total_mass,
            config,
            coefficients,
            coordinates: system.coordinates.clone(),
            masses: system.masses.clone(),
            velocities,
            kinetic_energy: 0.0,
            n_atoms: system.n_atoms,
            atomic_numbers: system.atomic_numbers.clone(),
            last_forces,
            friction,
            forces,
            energies,
            nonadiabatic_arr: nonad_arr.clone(),
            nonadiabatic_arr_old: nonad_arr,
            nonadiabatic_scalar: nonad_scalar.clone(),
            nonadiabatic_scalar_old: nonad_scalar,
            s_mat,
            dipole: dipole.clone(),
            dipole_old: dipole,
            saved_efactor: efactor,
            saved_p_rand,
            t_tot_last: None,
            thermostat,
        }
    }
}
