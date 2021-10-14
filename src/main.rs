#![allow(dead_code)]
#![allow(warnings)]

use clap::{App, Arg};
use env_logger::Builder;
use log::info;
use log::LevelFilter;
use std::io::Write;
use std::path::Path;
use std::process;
use std::time::{Duration, Instant};
use std::{env, fs};
use toml;

use crate::defaults::CONFIG_FILE_NAME;
use crate::initialization::io::{read_file_to_frame, Configuration};
use crate::initialization::Simulation;
use crate::initialization::SystemData;
use crate::interface::bagel::*;
use chemfiles::Frame;
use ndarray::Array2;
use crate::interface::QuantumChemistryInterface;

mod constants;
mod defaults;
mod dynamics;
mod initialization;
mod interface;
mod output;

#[macro_use]
extern crate clap;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

    let matches = App::new(crate_name!())
        .version(crate_version!())
        .about("software package for molecular dynamics using field-induced surface hopping")
        .arg(
            Arg::new("xyz-File")
                .about("Sets the xyz file to use")
                .required(true)
                .index(1),
        )
        .get_matches();

    let log_level: LevelFilter = match 0 {
        2 => LevelFilter::Trace,
        1 => LevelFilter::Debug,
        0 => LevelFilter::Info,
        -1 => LevelFilter::Warn,
        -2 => LevelFilter::Error,
        _ => LevelFilter::Info,
    };

    Builder::new()
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .filter(None, log_level)
        .init();

    // the file containing the cartesian coordinates is the only mandatory file to
    // start a calculation.
    let geometry_file = matches.value_of("xyz-File").unwrap();
    let frame: Frame = read_file_to_frame(geometry_file);

    // read fish configuration file, if it does not exist in the directory
    // the program initializes the default settings and writes a configuration file
    // to the directory
    let config_file_path: &Path = Path::new(CONFIG_FILE_NAME);
    let mut config_string: String = if config_file_path.exists() {
        fs::read_to_string(config_file_path).expect("Unable to read config file")
    } else {
        String::from("")
    };
    // load the configuration
    let config: Configuration = toml::from_str(&config_string).unwrap();
    // save the configuration file if it does not exist already so that the user can see
    // all the used options
    if config_file_path.exists() == false {
        config_string = toml::to_string(&config).unwrap();
        fs::write(config_file_path, config_string).expect("Unable to write config file");
    }

    // Generate system
    let system: SystemData = SystemData::from((frame, config.clone()));
    // Initialize dynamics
    let mut handler:Bagel_Handler = Bagel_Handler::from(&system);
    let mut dynamic: Simulation = Simulation::new(config, &system);
    // let mut dynamic: Simulation = Simulation::new(config, &system,Box::new(handler));
    dynamic.verlet_dynamics(&mut handler);
}
