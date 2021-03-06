use crate::initialization::Simulation;
use crate::interface::QuantumChemistryInterface;
use ::std::ops::AddAssign;
use ndarray::prelude::*;
use ndarray_linalg::c64;
use ndarray_npy::NpzWriter;
use std::fs::File;
use std::time::Instant;

impl Simulation {
    ///Ehrenfest dynamics routine of the struct Simulation
    pub fn ehrenfest_dynamics(&mut self, interface: &mut dyn QuantumChemistryInterface) {
        let mut npz = NpzWriter::new(File::create("arrays.npz").unwrap());
        let mut npz_c = NpzWriter::new(File::create("cis_arrays.npz").unwrap());
        let mut npz_q = NpzWriter::new(File::create("qtrans_arrays.npz").unwrap());
        let mut npz_mo = NpzWriter::new(File::create("mo_arrays.npz").unwrap());
        let mut npz_h = NpzWriter::new(File::create("h_arrays.npz").unwrap());
        let mut npz_x = NpzWriter::new(File::create("x_arrays.npz").unwrap());
        let mut npz_sign = NpzWriter::new(File::create("sign_arrays.npz").unwrap());
        let mut npz_sc = NpzWriter::new(File::create("sc_arrays.npz").unwrap());

        self.initialize_ehrenfest(
            interface,
            &mut npz,
            0,
            &mut npz_c,
            &mut npz_q,
            &mut npz_mo,
            &mut npz_h,
            &mut npz_x,
            &mut npz_sign,
            &mut npz_sc,
        );
        for step in 1..self.config.nstep {
            self.ehrenfest_step(
                interface,
                &mut npz,
                step,
                &mut npz_c,
                &mut npz_q,
                &mut npz_mo,
                &mut npz_h,
                &mut npz_x,
                &mut npz_sign,
                &mut npz_sc,
            );
        }
        npz.finish().unwrap();
        npz_c.finish().unwrap();
        npz_q.finish().unwrap();
        npz_mo.finish().unwrap();
        npz_h.finish().unwrap();
        npz_x.finish().unwrap();
        npz_sign.finish().unwrap();
        npz_sc.finish().unwrap();
    }

    pub fn ehrenfest_step(
        &mut self,
        interface: &mut dyn QuantumChemistryInterface,
        npz: &mut NpzWriter<File>,
        step: usize,
        npz_c: &mut NpzWriter<File>,
        npz_q: &mut NpzWriter<File>,
        npz_mo: &mut NpzWriter<File>,
        npz_h: &mut NpzWriter<File>,
        npz_x: &mut NpzWriter<File>,
        npz_sign: &mut NpzWriter<File>,
        npz_sc: &mut NpzWriter<File>,
    ) {
        let old_forces: Array2<f64> = self.forces.clone();
        let old_energy: f64 = self.energies[self.state] + self.kinetic_energy;
        // calculate the gradient and the excitonic couplings
        let excitonic_couplings: Array2<f64> = self.get_ehrenfest_data(
            interface, npz, npz_c, npz_q, npz_mo, npz_h, npz_x, npz_sign, npz_sc, step,
        );
        // convert to complex array
        let excitonic_couplings: Array2<c64> =
            excitonic_couplings.map(|val| val * c64::new(1.0, 0.0));

        // ehrenfest procedure
        self.coefficients = self.ehrenfest_rk(excitonic_couplings.view());
        // self.coefficients = self.ehrenfest_sod_integration(excitonic_couplings.view());
        // self.coefficients = self.ehrenfest_matrix_exponential_nacme(excitonic_couplings.view());

        // Calculate new coordinates from velocity-verlet
        self.velocities = self.get_velocities_verlet(old_forces.view());
        // remove tranlation and rotation from the velocities
        self.velocities = self.eliminate_translation_rotation_from_velocity();

        // calculate the kinetic energy
        self.kinetic_energy = self.get_kinetic_energy();

        // scale velocities
        self.velocities = self
            .thermostat
            .scale_velocities(self.velocities.view(), self.kinetic_energy);

        // Print settings
        self.print_data(Some(old_energy));

        // Calculate new coordinates from velocity-verlet
        self.coordinates = self.get_coord_verlet();

        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();
    }

    pub fn initialize_ehrenfest(
        &mut self,
        interface: &mut dyn QuantumChemistryInterface,
        npz: &mut NpzWriter<File>,
        step: usize,
        npz_c: &mut NpzWriter<File>,
        npz_q: &mut NpzWriter<File>,
        npz_mo: &mut NpzWriter<File>,
        npz_h: &mut NpzWriter<File>,
        npz_x: &mut NpzWriter<File>,
        npz_sign: &mut NpzWriter<File>,
        npz_sc: &mut NpzWriter<File>,
    ) {
        // remove COM from coordinates
        self.coordinates = self.shift_to_center_of_mass();
        self.initial_coordinates = self.coordinates.clone();
        // remove tranlation and rotation
        self.velocities = self.eliminate_translation_rotation_from_velocity();
        // do the first calculation using the QuantumChemistryInterface
        self.get_ehrenfest_data(
            interface, npz, npz_c, npz_q, npz_mo, npz_h, npz_x, npz_sign, npz_sc, step,
        );

        // calculate the kinetic energy
        self.kinetic_energy = self.get_kinetic_energy();

        // Print settings
        self.print_data(None);

        // Calculate new coordinates from velocity-verlet
        self.coordinates = self.get_coord_verlet();
        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();
    }

    pub fn get_ehrenfest_data(
        &mut self,
        interface: &mut dyn QuantumChemistryInterface,
        npz: &mut NpzWriter<File>,
        npz_c: &mut NpzWriter<File>,
        npz_q: &mut NpzWriter<File>,
        npz_mo: &mut NpzWriter<File>,
        npz_h: &mut NpzWriter<File>,
        npz_x: &mut NpzWriter<File>,
        npz_sign: &mut NpzWriter<File>,
        npz_sc: &mut NpzWriter<File>,
        step: usize,
    ) -> Array2<f64> {
        let abs_coefficients: Array1<f64> = self.coefficients.map(|val| val.norm());
        let tmp: (
            f64,
            Array2<f64>,
            Array2<f64>,
            Array2<f64>,
            Vec<Array2<f64>>,
            Vec<Array1<f64>>,
            Vec<Array2<f64>>,
            Vec<Array2<f64>>,
            Vec<Array2<f64>>,
            Array1<f64>,
            Array1<f64>,
        ) = interface.compute_ehrenfest(
            self.coordinates.view(),
            abs_coefficients.view(),
            self.config.ehrenfest_config.state_threshold,
            self.config.stepsize,
        );
        self.energies[0] = tmp.0;
        self.energies
            .slice_mut(s![1..])
            .assign(&(&tmp.2.diag().slice(s![1..]) + tmp.0));
        // let forces: Array2<f64> = tmp.0;

        let forces: Array2<f64> = tmp.1;
        for (idx, mass) in self.masses.iter().enumerate() {
            self.forces
                .slice_mut(s![idx, ..])
                .assign(&(-1.0 * &forces.slice(s![idx, ..]) / mass.to_owned()));
        }
        // update the nonadiabatic coupling
        // self.nonadiabatic_scalar = tmp.3;

        if self.config.ehrenfest_config.use_restraint {
            self.apply_harmonic_restraint();
        }
        npz.add_array(step.to_string(), &tmp.2).unwrap();
        npz_sign.add_array(step.to_string(), &tmp.9).unwrap();
        npz_sc.add_array(step.to_string(), &tmp.10).unwrap();
        for (idx, arr) in tmp.4.iter().enumerate() {
            let mut string: String = step.to_string();
            string.push_str(&String::from("-"));
            string.push_str(&idx.to_string());
            npz_c.add_array(string, arr).unwrap();
        }
        for (idx, arr) in tmp.5.iter().enumerate() {
            let mut string: String = step.to_string();
            string.push_str(&String::from("-"));
            string.push_str(&idx.to_string());
            npz_q.add_array(string, arr).unwrap();
        }
        for (idx, arr) in tmp.6.iter().enumerate() {
            let mut string: String = step.to_string();
            string.push_str(&String::from("-"));
            string.push_str(&idx.to_string());
            npz_mo.add_array(string, arr).unwrap();
        }
        for (idx, arr) in tmp.7.iter().enumerate() {
            let mut string: String = step.to_string();
            string.push_str(&String::from("-"));
            string.push_str(&idx.to_string());
            npz_h.add_array(string, arr).unwrap();
        }
        for (idx, arr) in tmp.8.iter().enumerate() {
            let mut string: String = step.to_string();
            string.push_str(&String::from("-"));
            string.push_str(&idx.to_string());
            npz_x.add_array(string, arr).unwrap();
        }

        // diabatic_couplings
        tmp.2
    }

    pub fn apply_harmonic_restraint(&mut self) {
        // calculate the force constant in au. The value of the config is in kcal/mol
        let force_constant: f64 = self.config.ehrenfest_config.force_constant * 0.00159362;
        // calculate the forces for the deviation from the initial coordinates
        let forces: Array2<f64> = -force_constant * (&self.coordinates - &self.initial_coordinates);

        for (idx, force) in forces.outer_iter().enumerate() {
            self.forces.slice_mut(s![idx, ..]).add_assign(&force);
        }
    }
}
