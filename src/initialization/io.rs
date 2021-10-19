#![allow(dead_code)]
#![allow(warnings)]
#[macro_use]
use clap::crate_version;
use crate::constants;
use crate::defaults::*;
use chemfiles::{Frame, Trajectory};
use clap::App;
use log::{debug, error, info, trace, warn};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::Path;
use std::ptr::eq;
use std::{env, fs};

// TODO: CHANGE STRINGS TO ENUMS

fn default_charge() -> i8 {
    CHARGE
}
fn default_verbose() -> i8 {
    VERBOSE
}
fn default_nstep() -> usize {
    NSTEP
}
fn default_stepsize() -> f64 {
    STEPSIZE
}
fn default_n_small_steps() -> usize {
    N_SMALL_STEPS
}
fn default_dyn_mode() -> char {
    DYN_MODE
}
fn default_temperature() -> f64 {
    TEMPERATURE
}
fn default_friction() -> f64 {
    FRICTION
}
fn default_inputflag() -> String {
    String::from("new")
}
fn default_print_coupling() -> bool {
    PRINT_COUPLING
}
fn default_print_coefficients() -> u8 {
    PRINT_COEFFICIENTS
}
fn default_nuclear_propagation() -> char {
    NUCLEAR_PROPAGATION
}
fn default_initial_state() -> usize {
    INITIAL_STATE
}
fn default_nstates() -> usize {
    NSTATES
}
fn default_states_reduced() -> bool {
    STATES_REDUCED
}
fn default_coupling() -> i8 {
    COUPLING
}
fn default_fieldflag() -> u8 {
    FIELDFLAG
}
fn default_integration_type() -> String {
    String::from("LD")
}
fn default_rescale_type() -> String {
    String::from("uniform")
}
fn default_scalar_coupling_treshold() -> f64 {
    SCALAR_COUPLING_TRESHOLD
}
fn default_force_switch_to_gs() -> bool {
    FORCE_SWITCH_TO_GS
}
fn default_artificial_energy_conservation() -> bool {
    ARTIFICIAL_ENERGY_CONSERVATION
}
fn default_extrapolate_forces() -> bool {
    EXTP
}
fn default_gs_dynamic() -> bool {
    GS_DYNAMIC
}
fn default_start_econst() -> f64 {
    START_ECONST * constants::FS_TO_AU
}
fn default_decoherence_correction() -> bool {
    DECOHERENCE_CORRECTION
}
fn default_time_coupling() -> f64 {
    TIME_COUPLING
}
fn default_velocity_generation() -> u8 {
    VELOCITY_GENERATION
}

fn default_number_pulses() -> usize {
    1
}
fn default_pulse_param_1() -> f64 {
    0.1
}
fn default_pulse_param_2() -> f64 {
    0.1
}
fn default_pulse_param_3() -> f64 {
    0.1
}
fn default_pulse_param_4() -> f64 {
    0.1
}
fn default_pulse_param_5() -> f64 {
    0.1
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DynamicConfiguration {
    #[serde(default = "default_verbose")]
    pub verbose: i8,
    #[serde(default = "default_nstep")]
    pub nstep: usize,
    #[serde(default = "default_stepsize")]
    pub stepsize: f64,
    #[serde(default = "default_n_small_steps")]
    pub n_small_steps: usize,
    #[serde(default = "default_dyn_mode")]
    pub dyn_mode: char,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_friction")]
    pub friction: f64,
    #[serde(default = "default_inputflag")]
    pub inputflag: String,
    #[serde(default = "default_print_coupling")]
    pub print_coupling: bool,
    #[serde(default = "default_print_coefficients")]
    pub print_coefficients: u8,
    #[serde(default = "default_nuclear_propagation")]
    pub nuclear_propagation: char,
    #[serde(default = "default_charge")]
    pub charge: i8,
    #[serde(default = "default_initial_state")]
    pub initial_state: usize,
    #[serde(default = "default_nstates")]
    pub nstates: usize,
    #[serde(default = "default_states_reduced")]
    pub states_reduced: bool,
    #[serde(default = "default_coupling")]
    pub coupling: i8,
    #[serde(default = "default_fieldflag")]
    pub fieldflag: u8,
    #[serde(default = "default_integration_type")]
    pub integration_type: String,
    #[serde(default = "default_rescale_type")]
    pub rescale_type: String,
    #[serde(default = "default_scalar_coupling_treshold")]
    pub scalar_coupling_treshold: f64,
    #[serde(default = "default_force_switch_to_gs")]
    pub force_switch_to_gs: bool,
    #[serde(default = "default_artificial_energy_conservation")]
    pub artificial_energy_conservation: bool,
    #[serde(default = "default_extrapolate_forces")]
    pub extrapolate_forces: bool,
    #[serde(default = "default_gs_dynamic")]
    pub gs_dynamic: bool,
    #[serde(default = "default_start_econst")]
    pub start_econst: f64,
    #[serde(default = "default_decoherence_correction")]
    pub decoherence_correction: bool,
    #[serde(default = "default_time_coupling")]
    pub time_coupling: f64,
    #[serde(default = "default_velocity_generation")]
    pub velocity_generation: u8,
}

impl DynamicConfiguration {
    pub fn new() -> Self {
        // read tincr configuration file, if it does not exist in the directory
        // the program initializes the default settings and writes an configuration file
        // to the directory
        let config_file_path: &Path = Path::new(CONFIG_FILE_NAME);
        let mut config_string: String = if config_file_path.exists() {
            fs::read_to_string(config_file_path).expect("Unable to read config file")
        } else {
            String::from("")
        };
        // load the configration settings
        let config: Self = toml::from_str(&config_string).unwrap();
        // save the configuration file if it does not exist already
        if config_file_path.exists() == false {
            config_string = toml::to_string(&config).unwrap();
            fs::write(config_file_path, config_string).expect("Unable to write config file");
        }
        return config;
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Pulse_Configuration {
    #[serde(default = "default_number_pulses")]
    pub number_pulses: usize,
    #[serde(default = "default_pulse_param_1")]
    pub pulse_param_1: f64,
    #[serde(default = "default_pulse_param_2")]
    pub pulse_param_2: f64,
    #[serde(default = "default_pulse_param_3")]
    pub pulse_param_3: f64,
    #[serde(default = "default_pulse_param_4")]
    pub pulse_param_4: f64,
}

impl Pulse_Configuration {
    pub fn new() -> Self {
        let config_file_path: &Path = Path::new(PULSE_CONFIG);
        let mut config_string: String = if config_file_path.exists() {
            fs::read_to_string(config_file_path).expect("Unable to read config file")
        } else {
            String::from("")
        };
        // load the configration settings
        let config: Self = toml::from_str(&config_string).unwrap();
        // save the configuration file if it does not exist already
        if config_file_path.exists() == false {
            config_string = toml::to_string(&config).unwrap();
            fs::write(config_file_path, config_string).expect("Unable to write config file");
        }
        return config;
    }
}

/// Read a xyz-geometry file like .xyz or .pdb and returns a [Frame](chemfiles::Frame)
pub fn read_file_to_frame(filename: &str) -> Frame {
    // read the geometry file
    let mut trajectory = Trajectory::open(filename, 'r').unwrap();
    let mut frame = Frame::new();
    // if multiple geometries are contained in the file, we will only use the first one
    trajectory.read(&mut frame).unwrap();
    return frame;
}

/// Extract the atomic numbers and positions (in bohr) from a [Frame](chemfiles::frame)
pub fn frame_to_coordinates(frame: Frame) -> (Vec<u8>, Array2<f64>) {
    let mut positions: Array2<f64> = Array2::from_shape_vec(
        (frame.size() as usize, 3),
        frame
            .positions()
            .iter()
            .flat_map(|array| array.iter())
            .cloned()
            .collect(),
    )
    .unwrap();
    // transform the coordinates from angstrom to bohr
    positions = positions / constants::BOHR_TO_ANGS;
    // read the atomic number of each coordinate
    let atomic_numbers: Vec<u8> = (0..frame.size() as u64)
        .map(|i | frame.atom(i as usize).atomic_number() as u8)
        .collect();

    // let mut smiles_repr: Trajectory = Trajectory::memory_writer("SMI").unwrap();
    // smiles_repr.write(&mut frame).unwrap();
    // let smiles: String = smiles_repr.memory_buffer().unwrap().replace('~', "").replace('\n', "");
    return (atomic_numbers, positions);
}
