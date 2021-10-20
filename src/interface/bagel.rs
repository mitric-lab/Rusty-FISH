use crate::constants;
use crate::constants::ATOM_NAMES;
use crate::defaults::{BAGEL_EDIT_NAME, BAGEL_FILE_NAME};
use hashbrown::HashMap;
use ndarray::prelude::*;
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Result, Value};
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Output};
use std::{env, fs};
use crate::initialization::SystemData;

#[derive(Clone)]
pub struct Bagel_Handler {
    pub atomic_numbers: Vec<u8>,
    pub masses: Array1<f64>,
    pub nstates: usize,
    pub state: usize,
    pub n_atoms: usize,
    pub gs_dynamic: bool,
    pub coordinates: Array2<f64>,
    pub forces: Array2<f64>,
    pub energies: Array1<f64>,
    pub nad_coupling: Array3<f64>,
    pub transition_dipoles: Array3<f64>,
}

impl From<&SystemData> for Bagel_Handler{
    fn from(data:&SystemData) -> Self {
        let nstates = data.config.nstates;
        Self::new(
            &data.atomic_numbers,
            data.masses.view(),
            nstates,data.config.gs_dynamic,
            data.coordinates.view(),
            data.config.initial_state
        )
    }
}

impl Bagel_Handler {
    pub fn new(
        atomic_numbers: &Vec<u8>,
        masses: ArrayView1<f64>,
        nstates: usize,
        gs_dynamic: bool,
        coordinates: ArrayView2<f64>,
        state: usize,
    ) -> Bagel_Handler {
        let n_atoms: usize = atomic_numbers.len();
        let energies: Array1<f64> = Array1::zeros(nstates);
        let forces: Array2<f64> = Array2::zeros(coordinates.raw_dim());
        let nad_coupling: Array3<f64> = Array3::zeros((nstates, n_atoms, 3));
        let new_coords: Array2<f64> = coordinates.to_owned() * constants::BOHR_TO_ANGS;
        let dipoles: Array3<f64> = Array3::zeros((nstates, nstates, 3));

        Bagel_Handler {
            atomic_numbers: atomic_numbers.clone(),
            masses: masses.to_owned(),
            nstates: nstates,
            n_atoms: n_atoms,
            gs_dynamic: gs_dynamic,
            coordinates: new_coords,
            forces: forces,
            energies: energies,
            state: state,
            nad_coupling: nad_coupling,
            transition_dipoles: dipoles,
        }
    }

    pub fn write_bagel_file(&self) {
        // read string from file
        let bagel_file_path: &Path = Path::new(BAGEL_FILE_NAME);
        let mut bagel_string: String = if bagel_file_path.exists() {
            fs::read_to_string(bagel_file_path).expect("Unable to read config file")
        } else {
            String::from("")
        };

        // safe as json value
        let mut v: Value = serde_json::from_str(&bagel_string).unwrap();

        // change geometry of template
        let mut string: String = String::from("[");
        for (index, atom) in self.atomic_numbers.iter().enumerate() {
            let mut first_part = String::from("\n\t\t{ \"atom\" : \"");
            let atom = ATOM_NAMES[*atom as usize];
            first_part.push_str(atom);
            let second_part = String::from("\", \"xyz\" : ");
            first_part.push_str(&second_part);
            // let coords = self.coordinates.slice(s![index, ..]).to_string();
            let str: String = format!("{:12.10}", self.coordinates.slice(s![index, ..]));
            // first_part.push_str(&format!("{:12.10}",coords));
            first_part.push_str(&str);
            if index == self.atomic_numbers.len() - 1 {
                let last_part = String::from("}");
                first_part.push_str(&last_part);
            } else {
                let last_part = String::from("},");
                first_part.push_str(&last_part);
            }
            string.push_str(&first_part);
        }
        string.push_str(&String::from("]"));

        let new_value: Value = serde_json::from_str(&string).unwrap();
        v["bagel"][0]["geometry"] = new_value;

        // v["bagel"][0]["geometry"];
        // let mut input: Bagel_Json = serde_json::from_str(&bagel_string).unwrap();
        // let mut geometry_vec:Vec<GeometryEntry> = Vec::new();
        // // fill geometry of template
        // for (index, atom) in self.atomic_numbers.iter().enumerate() {
        //     let entry: GeometryEntry = GeometryEntry::new(
        //         String::from(ATOM_NAMES[*atom as usize]),
        //         coordinates.slice(s![index, ..]).to_vec(),
        //     );
        //     geometry_vec.push(entry);
        // }
        // // replace geometry of the template
        // input.bagel.geom_block.geometry = geometry_vec;

        // println!("New Geometry {:?}",input.bagel.geom_block.geometry);

        let bagel_string: String = v.to_string();
        let bagel_file_path: &Path = Path::new(BAGEL_EDIT_NAME);
        fs::write(bagel_file_path, bagel_string).expect("Unable to write config file");
    }

    pub fn run_bagel(&self) {
        self.write_bagel_file();

        let mut run: Command = Command::new("BAGEL");
        run.arg("bagel_edit.json");
        let out = run.output().expect("Unable to run bagel!");
        let string: String = String::from("Test");
        fs::write("bagel.out", &out.stdout).expect("Unable to write file bagel.out");
    }

    pub fn read_energies(&mut self) {
        let bagel_file_path: &Path = Path::new("ENERGY.out");
        let mut bagel_string: String = if bagel_file_path.exists() {
            fs::read_to_string(bagel_file_path).expect("Unable to read config file")
        } else {
            String::from("")
        };
        let strings: Vec<&str> = bagel_string.split("\n").collect();
        for state in (0..self.nstates) {
            let str: &str = strings[state].trim();
            self.energies[state] = str.parse::<f64>().unwrap();
        }
    }

    pub fn read_gradients(&mut self) {
        let mut filename: String = String::from("FORCE_");
        filename.push_str(&self.state.to_string());
        filename.push_str(&String::from(".out"));

        let bagel_file_path: &Path = Path::new(&filename);
        let mut bagel_string: String = if bagel_file_path.exists() {
            fs::read_to_string(bagel_file_path).expect("Unable to read config file")
        } else {
            String::from("")
        };
        let mut strings: Vec<&str> = bagel_string.split("\n").collect();
        strings.remove(0);

        for atom in (0..self.n_atoms) {
            let str: Vec<&str> = strings[atom].split(" ").collect();
            let mut vec: Vec<f64> = Vec::new();
            for item in (0..str.len()) {
                if str[item].len() > 0 {
                    // vec.push(-str[item].parse::<f64>().unwrap() / self.masses[atom]);
                    vec.push(str[item].parse::<f64>().unwrap());
                }
            }
            vec.remove(0);
            self.forces
                .slice_mut(s![atom, ..])
                .assign(&Array::from(vec));
        }
    }

    pub fn read_nonadiabatic_coupling(&mut self) {
        let mut iter: usize = 0;
        for state_i in (0..self.nstates) {
            for state_j in (0..state_i) {
                let mut nad: Array2<f64> = Array2::zeros((self.n_atoms, 3));
                let mut filename: String = String::from("NACME_");
                filename.push_str(&state_j.to_string());
                filename.push_str(&String::from("_"));
                filename.push_str(&state_i.to_string());
                filename.push_str(&String::from(".out"));

                let bagel_file_path: &Path = Path::new(&filename);
                let mut bagel_string: String = if bagel_file_path.exists() {
                    fs::read_to_string(bagel_file_path).expect("Unable to read config file")
                } else {
                    String::from("")
                };
                let mut strings: Vec<&str> = bagel_string.split("\n").collect();
                strings.remove(0);

                for atom in (0..self.n_atoms) {
                    let str: Vec<&str> = strings[atom].split(" ").collect();
                    let mut vec: Vec<f64> = Vec::new();
                    for item in (0..str.len()) {
                        if str[item].len() > 0 {
                            vec.push(str[item].parse::<f64>().unwrap());
                        }
                    }
                    vec.remove(0);
                    nad.slice_mut(s![atom, ..]).assign(&Array::from(vec));
                }
                self.nad_coupling.slice_mut(s![iter, .., ..]).assign(&nad);
                iter += 1;
            }
        }
    }

    pub fn read_transition_dipoles(&mut self) {
        let bagel_file_path: &Path = Path::new("bagel.out");
        let mut bagel_string: String = if bagel_file_path.exists() {
            fs::read_to_string(bagel_file_path).expect("Unable to read config file")
        } else {
            String::from("")
        };
        let strings: Vec<&str> = bagel_string.split_inclusive("\n").collect();
        for state_i in (1..self.nstates) {
            for state_j in (0..state_i) {
                let mut string: String = String::from("Transition    ");
                string.push_str(&state_i.to_string());
                string.push_str(&String::from(" - "));
                string.push_str(&state_j.to_string());

                let mut dipole_vec: Vec<f64> = Vec::new();

                for elem in strings.iter() {
                    let ind: Option<usize> = elem.find(&string);
                    if ind.clone().is_some() {
                        let str: Vec<&str> = elem.split(":").collect();
                        let last_str: Vec<&str> = str[1].split(" ").collect();
                        for item in last_str.iter() {
                            if !item.contains("\n") && item.len() > 1 {
                                if item.contains(",") {
                                    let val: String = item.replace(",", "");
                                    dipole_vec.push(val.parse::<f64>().unwrap());
                                } else if item.contains(")") {
                                    let val: String = item.replace(")", "");
                                    dipole_vec.push(val.parse::<f64>().unwrap());
                                }
                            }
                        }
                    }
                }
                self.transition_dipoles
                    .slice_mut(s![state_j, state_i, ..])
                    .assign(&Array::from(dipole_vec.clone()));
                self.transition_dipoles
                    .slice_mut(s![state_i, state_j, ..])
                    .assign(&Array::from(dipole_vec));
            }
        }
        // let transition_clone:Array3<f64> = self.transition_dipoles.clone();
        // for i in (0..self.nstates){
        //     for j in (0..self.nstates){
        //         if i < j{
        //             self.transition_dipoles.slice_mut(s![j,i,..]).assign(&transition_clone.slice(s![i,j,..]));
        //         }
        //     }
        // }
    }

    pub fn get_all(
        &mut self,
        coordinates: ArrayView2<f64>,
        state: usize,
    ) -> (Array1<f64>, Array2<f64>, Array3<f64>, Array3<f64>) {
        self.coordinates = coordinates.to_owned() * constants::BOHR_TO_ANGS;
        self.state = state;
        self.run_bagel();
        self.read_energies();
        self.read_gradients();
        if self.gs_dynamic == false {
            self.read_transition_dipoles();
            self.read_nonadiabatic_coupling()
        }
        return (
            self.energies.clone(),
            self.forces.clone(),
            self.nad_coupling.clone(),
            self.transition_dipoles.clone(),
        );
    }
}

#[test]
pub fn test_read_all() {
    use crate::constants;

    let mut positions: Array2<f64> = array![
        [-14.223363827698291, 3.082634638057034, 0.0],
        [-11.699199951251911, 3.059806746471555, 0.0],
        [-15.293893677298792, 1.704986498682355, 1.07880685002636],
        [-15.268817011625007, 4.479406805812925, -1.07880685002636],
        [-10.650893280877012, 1.978807814140629, -1.3929360237229014],
        [-10.631523588099597, 4.121681650421268, 1.3929360237229014]
    ];

    let atomic_numbers: Vec<u8> = vec![6, 6, 1, 1, 1, 1];
    let n_atoms: usize = atomic_numbers.len();
    let mut masses: Array1<f64> = Array1::zeros(n_atoms);
    for atom in (0..n_atoms) {
        masses[atom] = constants::ATOMIC_MASSES[&atomic_numbers[atom]];
    }

    let mut bagel_handler: Bagel_Handler = Bagel_Handler::new(
        &atomic_numbers,
        masses.view(),
        3,
        false,
        positions.view(),
        0,
    );

    bagel_handler.read_energies();
    println!("Energies {}", bagel_handler.energies);
    bagel_handler.read_gradients();
    println!("Forces {}", bagel_handler.forces);
    bagel_handler.read_nonadiabatic_coupling();
    println!("Nonadiabatic Coupling {}", bagel_handler.nad_coupling);
    bagel_handler.read_transition_dipoles();
    println!("Transition dipoles {}", bagel_handler.transition_dipoles);

    assert!(1 == 2);
}

//
// #[derive(Serialize, Deserialize, Clone)]
// pub struct Bagel_Json {
//     pub bagel: Bagel_Json_2,
// }
// #[derive(Serialize, Deserialize, Clone)]
// pub struct Bagel_Json_2 {
//     pub geom_block: BagelConfig,
//     pub method_block: BagelBlock2,
// }
// #[derive(Serialize, Deserialize, Clone)]
// pub struct BagelBlock2 {
//     pub title: String,
// }
//
// #[derive(Serialize, Deserialize, Clone)]
// pub struct BagelConfig {
//     pub title: String,
//     pub basis: String,
//     pub df_basis: String,
//     pub angstrom: String,
//     pub geometry: Vec<GeometryEntry>,
// }
// #[derive(Serialize, Deserialize, Clone, Debug)]
// pub struct GeometryEntry {
//     pub atom: String,
//     pub xyz: Vec<f64>,
// }
//
// impl GeometryEntry {
//     fn new(string: String, xyz: Vec<f64>) -> GeometryEntry {
//         GeometryEntry {
//             atom: string,
//             xyz: xyz,
//         }
//     }
// }
//
// #[test]
// fn test_bagel_read() {
//     let atomic_numbers: Vec<u8> = vec![1, 1];
//     let masses: Array1<f64> = array![1.0, 1.0];
//     let coordinates:Array2<f64> = array![[0.0,0.0,1.0],[0.0,0.0,-1.0]];
//     let n_states: usize = 1;
//     let gs_dynamic: bool = true;
//     let handler: Bagel_Handler =
//         Bagel_Handler::new(&atomic_numbers, masses.view(), n_states, gs_dynamic);
//
//     handler.write_bagel_file(coordinates.view());
//
//     assert!(1 == 2);
// }
// #[test]
// fn test_bagel_2() {
//     let bagel_file_path: &Path = Path::new(BAGEL_FILE_NAME);
//     let mut bagel_string: String = if bagel_file_path.exists() {
//         fs::read_to_string(bagel_file_path).expect("Unable to read config file")
//     } else {
//         String::from("")
//     };
//
//     let mut v: Value = serde_json::from_str(&bagel_string).unwrap();
//     println!("{:?}", v["bagel"][0]["geometry"]);
//     println!(" ");
//
//     let atomic_numbers: Vec<u8> = vec![1, 1];
//     let coordinates:Array2<f64> = array![[0.5,0.5,1.0],[0.2,0.2,-1.0]];
//
//     let mut string:String = String::from("[");
//     for (index,atom) in atomic_numbers.iter().enumerate(){
//         let mut first_part = String::from("\n\t\t{ \"atom\" : \"");
//         let atom = ATOM_NAMES[*atom as usize];
//         first_part.push_str(atom);
//         let second_part = String::from("\", \"xyz\" : ");
//         first_part.push_str(&second_part);
//         let coords = coordinates.slice(s![index,..]).to_string();
//         first_part.push_str(&coords);
//         if index == atomic_numbers.len()-1{
//             let last_part = String::from("}");
//             first_part.push_str(&last_part);
//         }
//         else{
//             let last_part = String::from("},");
//             first_part.push_str(&last_part);
//         }
//         string.push_str(&first_part);
//     }
//     string.push_str(&String::from("]"));
//
//     let new_value: Value = serde_json::from_str(&string).unwrap();
//
//     println!("Newest string {}",string);
//     println!("New value {:?}", new_value);
//     println!(" ");
//
//     v["bagel"][0]["geometry"] = new_value;
//
//     println!("Changed v {:?}", v["bagel"][0]["geometry"]);
//     println!(" ");
//
//     let mut bagel: Bagel_Json = serde_json::from_str(&bagel_string).unwrap();
//     println!(
//         "Bagel json file geom test: {:?}",
//         bagel.bagel.geom_block.geometry
//     );
//
//     bagel.bagel.geom_block.geometry = Vec::new();
//     let new_entry_1: GeometryEntry = GeometryEntry::new(String::from("H"), vec![0.1, 0.1, 0.1]);
//     let new_entry_2: GeometryEntry = GeometryEntry::new(String::from("H"), vec![0.2, 0.2, 0.2]);
//     bagel.bagel.geom_block.geometry.push(new_entry_1);
//     bagel.bagel.geom_block.geometry.push(new_entry_2);
//
//     println!(
//         "Bagel json file new geom: {:?}",
//         bagel.bagel.geom_block.geometry
//     );
//
//     assert!(1 == 2);
// }
