use crate::constants;
use crate::initialization::{frame_to_coordinates, Configuration};
use chemfiles::Frame;
use hashbrown::HashMap;
use itertools::Itertools;
use ndarray::prelude::*;

pub struct SystemData {
    // Type that holds all the input settings from the user.
    pub config: Configuration,
    pub n_atoms: usize,
    pub atomic_numbers: Vec<u8>,
    pub coordinates: Array2<f64>,
    pub masses: Array1<f64>,
}

impl From<(Vec<u8>, Array2<f64>, Configuration)> for SystemData {
    fn from(molecule: (Vec<u8>, Array2<f64>, Configuration)) -> Self {
        let mut masses: Vec<f64> = Vec::new();
        molecule.0.iter().for_each(|num| {
            masses.push(constants::ATOMIC_MASSES[num]);
        });
        let masses: Array1<f64> = Array::from(masses);

        Self {
            config: molecule.2,
            n_atoms: molecule.0.len(),
            atomic_numbers: molecule.0,
            coordinates: molecule.1,
            masses: masses,
        }
    }
}

impl From<(Frame, Configuration)> for SystemData {
    /// Creates a new [SystemInput] from a [Frame](chemfiles::Frame) and
    /// the global configuration as [Configuration](crate::io::settings::Configuration).
    fn from(frame: (Frame, Configuration)) -> Self {
        let (numbers, coords) = frame_to_coordinates(frame.0);
        Self::from((numbers, coords, frame.1))
    }
}
