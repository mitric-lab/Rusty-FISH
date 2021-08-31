use crate::initialization::Simulation;
use crate::output::Restart_Output;
use crate::defaults;
use ndarray::prelude::*;
use std::path::Path;
use serde_yaml;
use std::fs;
use ndarray_linalg::c64;

// Read restart parameters from a file
pub fn read_restart_parameters()->(Array2<f64>,Array2<f64>,Array3<f64>,Array1<c64>){
    let restart_file_path: &Path = Path::new(defaults::RESTART_FILE_NAME);
    // check if file exists
    let mut restart_string: String = if restart_file_path.exists() {
        fs::read_to_string(restart_file_path).expect("Unable to read restart file")
    } else {
        String::from("")
    };
    // create Restart_Output struct
    let restart:Restart_Output = serde_yaml::from_str(&restart_string).unwrap();

    // take arrays from restart
    let coordinates:Array2<f64> = restart.coordinates;
    let velocities:Array2<f64> = restart.velocities;
    let nonadiabatic_arr:Array3<f64> = restart.nonadiabatic_arr;
    let coefficients:Array1<c64> = restart.coefficients;

    return (coordinates, velocities, nonadiabatic_arr,coefficients);
}