use clap::{App, Arg};
use env_logger::Builder;
use log::LevelFilter;
use std::io::Write;
use std::path::Path;
use std::{env, fs};
use chemfiles::Frame;
use crate::defaults::CONFIG_FILE_NAME;
use crate::initialization::io::{read_file_to_frame, DynamicConfiguration};
use crate::initialization::SystemData;
mod constants;
mod defaults;
mod dynamics;
mod initialization;
mod interface;
mod output;
#[macro_use]
extern crate clap;

fn main() {
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
    let config: DynamicConfiguration = toml::from_str(&config_string).unwrap();
    // save the configuration file if it does not exist already so that the user can see
    // all the used options
    if !config_file_path.exists() {
        config_string = toml::to_string(&config).unwrap();
        fs::write(config_file_path, config_string).expect("Unable to write config file");
    }

    // Generate system
    let _system: SystemData = SystemData::from((frame, config));
}
