use crate::constants;
use ndarray::prelude::*;
use ndarray::{Array2, ArrayView2};
use ndarray_linalg::c64;
use serde::{Deserialize, Serialize};
use serde_json;
use serde_yaml;
use std::fs;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::Command;
use toml;

#[derive(Serialize, Deserialize, Clone)]
pub struct Standard_Output {
    pub time: f64,
    pub coordinates: Array2<f64>,
    pub velocities: Array2<f64>,
    pub kinetic_energy: f64,
    pub electronic_energy: f64,
    pub total_energy: f64,
    pub energy_difference: f64,
    pub forces: Array2<f64>,
    pub state: usize,
}

impl Standard_Output {
    pub fn new(
        time: f64,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        kinetic_energy: f64,
        electronic_energy: f64,
        total_energy: f64,
        energy_difference: f64,
        forces: ArrayView2<f64>,
        state: usize,
    ) -> Standard_Output {
        let time: f64 = time / constants::FS_TO_AU;
        Standard_Output {
            time: time,
            coordinates: coordinates.to_owned() * constants::BOHR_TO_ANGS,
            velocities: velocities.to_owned(),
            kinetic_energy: kinetic_energy,
            electronic_energy: electronic_energy,
            total_energy: total_energy,
            energy_difference: energy_difference,
            forces: forces.to_owned(),
            state: state,
        }
    }
}
#[derive(Serialize, Deserialize, Clone)]
pub struct XYZ_Output {
    pub n_atoms: usize,
    pub coordinates: Array2<f64>,
    pub atomic_numbers: Vec<u8>,
}

impl XYZ_Output {
    pub fn new(
        n_atoms: usize,
        coordinates: ArrayView2<f64>,
        atomic_numbers: Vec<u8>,
    ) -> XYZ_Output {
        XYZ_Output {
            n_atoms: n_atoms,
            coordinates: coordinates.to_owned() * constants::BOHR_TO_ANGS,
            atomic_numbers: atomic_numbers,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct hopping_output {
    pub time: f64,
    pub coefficients_real: Array1<f64>,
    pub coefficients_imag: Array1<f64>,
    pub nonadiabatic_scalar: Array2<f64>,
    pub dipoles: Array3<f64>,
}

impl hopping_output {
    pub fn new(
        time: f64,
        coefficients: ArrayView1<c64>,
        nonadiabatic_scalar: ArrayView2<f64>,
        dipoles: ArrayView3<f64>,
    ) -> hopping_output {
        let time: f64 = time / constants::FS_TO_AU;
        let coefficients_real: Vec<f64> = coefficients.iter().map(|val| val.re).collect();
        let coefficients_imag: Vec<f64> = coefficients.iter().map(|val| val.im).collect();
        hopping_output {
            time: time,
            coefficients_real: Array::from(coefficients_real),
            coefficients_imag: Array::from(coefficients_imag),
            nonadiabatic_scalar: nonadiabatic_scalar.to_owned(),
            dipoles: dipoles.to_owned(),
        }
    }
}

// TODO: save nonadiabatic couplings
#[derive(Serialize, Deserialize, Clone)]
pub struct Restart_Output {
    pub n_atoms: usize,
    pub coordinates: Array2<f64>,
    pub velocities: Array2<f64>,
    pub nonadiabatic_arr: Array3<f64>,
    pub coefficients: Array1<c64>,
}

impl Restart_Output {
    pub fn new(
        n_atoms: usize,
        coordinates: ArrayView2<f64>,
        velocities: ArrayView2<f64>,
        nonadiabatic_arr: ArrayView3<f64>,
        coefficients: ArrayView1<c64>,
    ) -> Restart_Output {
        Restart_Output {
            n_atoms: n_atoms,
            coordinates: coordinates.to_owned(),
            velocities: velocities.to_owned(),
            nonadiabatic_arr: nonadiabatic_arr.to_owned(),
            coefficients: coefficients.to_owned(),
        }
    }
}

pub fn write_full(standard: &Standard_Output) {
    let file_path: &Path = Path::new("dynamics.out");
    let full: String = serde_yaml::to_string(standard).unwrap();
    if file_path.exists() {
        let mut file = OpenOptions::new().append(true).open(file_path).unwrap();
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", full)).unwrap();
        stream.flush().unwrap();
        // write!(&mut file,full);
    } else {
        fs::write(file_path, full).expect("Unable to write to dynamics.out file");
    }
}

pub fn write_full_custom(standard: &Standard_Output, masses: ArrayView1<f64>) {
    let mut string: String = String::from("######################################\n");
    string.push_str(&String::from("time: "));
    string.push_str(&standard.time.to_string());
    string.push_str(&String::from("\n[coordinates]\n"));
    let n_atoms: usize = standard.coordinates.dim().0;
    for atom in (0..n_atoms) {
        for item in (0..3) {
            let str: String = standard.coordinates.slice(s![atom, item]).to_string();
            string.push_str(&str);
            string.push_str("\t");
        }
        string.push_str("\n");
    }
    string.push_str(&String::from("[velocities]\n"));
    for atom in (0..n_atoms) {
        for item in (0..3) {
            let str: String = standard.velocities.slice(s![atom, item]).to_string();
            string.push_str(&str);
            string.push_str("\t");
        }
        string.push_str("\n");
    }
    string.push_str("Kinetic Energy: ");
    string.push_str(&standard.kinetic_energy.to_string());
    string.push_str("\n");
    string.push_str("Electronic energy: ");
    string.push_str(&standard.electronic_energy.to_string());
    string.push_str("\n");
    string.push_str("Total energy: ");
    string.push_str(&standard.total_energy.to_string());
    string.push_str("\n");
    string.push_str("Energy difference ");
    string.push_str(&standard.energy_difference.to_string());
    string.push_str("\n");
    string.push_str("Electronic state: ");
    string.push_str(&standard.state.to_string());
    string.push_str("\n");
    string.push_str("[forces]\n");
    for atom in (0..n_atoms) {
        for item in (0..3) {
            let str: String = (standard.forces[[atom, item]] * masses[atom]).to_string();
            string.push_str(&str);
            string.push_str("\t");
        }
        string.push_str("\n");
    }

    let file_path: &Path = Path::new("dynamics.out");
    if file_path.exists() {
        let mut file = OpenOptions::new().append(true).open(file_path).unwrap();
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
        // write!(&mut file,full);
    } else {
        fs::write(file_path, string).expect("Unable to write to dynamics.out file");
    }
}

pub fn write_xyz(xyz: &XYZ_Output) {
    let file_path: &Path = Path::new("dynamics.xyz");
    let xyz: String = serde_yaml::to_string(xyz).unwrap();
    if file_path.exists() {
        let mut file = OpenOptions::new().append(true).open(file_path).unwrap();
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", xyz)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, xyz).expect("Unable to write to dynamics.xyz file");
    }
}
pub fn write_xyz_custom(xyz: &XYZ_Output) {
    let file_path: &Path = Path::new("dynamics.xyz");
    let mut string: String = xyz.n_atoms.to_string();
    string.push_str("\n");
    string.push_str("\n");
    for atom in (0..xyz.n_atoms) {
        let str: String = constants::ATOM_NAMES[xyz.atomic_numbers[atom] as usize].to_string();
        string.push_str(&str);
        string.push_str("\t");
        for item in (0..3) {
            let str: String = xyz.coordinates.slice(s![atom, item]).to_string();
            string.push_str(&str);
            string.push_str("\t");
        }
        string.push_str("\n");
    }

    if file_path.exists() {
        let mut file = OpenOptions::new().append(true).open(file_path).unwrap();
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to dynamics.xyz file");
    }
}

pub fn write_restart(restart: &Restart_Output) {
    let file_path: &Path = Path::new("dynamics_restart.out");
    let restart: String = serde_yaml::to_string(restart).unwrap();
    fs::write(file_path, restart).expect("Unable to write restart file");
}

pub fn write_restart_custom(restart: &Restart_Output) {
    let mut string: String = restart.n_atoms.to_string();
    string.push_str("\n");
    string.push_str("\n");
    for atom in (0..restart.n_atoms) {
        for item in (0..3) {
            let str: String = restart.coordinates.slice(s![atom, item]).to_string();
            string.push_str(&str);
            string.push_str("\t");
        }
        string.push_str("\n");
    }
    string.push_str("\n");
    for atom in (0..restart.n_atoms) {
        for item in (0..3) {
            let str: String = restart.velocities.slice(s![atom, item]).to_string();
            string.push_str(&str);
            string.push_str("\t");
        }
        string.push_str("\n");
    }
    let file_path: &Path = Path::new("dynamics_restart.out");
    fs::write(file_path, string).expect("Unable to write restart file");
}

pub fn write_hopping(hopping_out: &hopping_output) {
    let file_path: &Path = Path::new("hopping.dat");
    let mut hopp: String = String::from("#############################\n");
    hopp.push_str(&toml::to_string(hopping_out).unwrap());
    if file_path.exists() {
        let mut file = OpenOptions::new().append(true).open(file_path).unwrap();
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", hopp)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, hopp).expect("Unable to write to hopping.dat file");
    }
}

pub fn write_energies(energies: ArrayView1<f64>) {
    let file_path: &Path = Path::new("energies.dat");
    let mut string: String = String::from("");
    for (ind, energy) in energies.iter().enumerate() {
        // string.push_str(&energy.to_string());
        // string.push_str(&String::from("\t"));
        if ind == 0 {
            string.push_str(&energy.to_string());
            string.push_str(&String::from("\t"));
        } else {
            string.push_str(&(energies[0] - energy).abs().to_string());
            string.push_str(&String::from("\t"));
        }
    }
    string.push_str("\n");

    if file_path.exists() {
        let mut file = OpenOptions::new().append(true).open(file_path).unwrap();
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to energies.dat file");
    }
}

pub fn write_state(electronic_state: usize) {
    let file_path: &Path = Path::new("state.dat");
    let mut string: String = electronic_state.to_string();
    string.push_str(&String::from("\n"));

    if file_path.exists() {
        let mut file = OpenOptions::new().append(true).open(file_path).unwrap();
        let mut stream = BufWriter::new(file);
        stream.write_fmt(format_args!("{}", string)).unwrap();
        stream.flush().unwrap();
    } else {
        fs::write(file_path, string).expect("Unable to write to state.dat file");
    }
}
