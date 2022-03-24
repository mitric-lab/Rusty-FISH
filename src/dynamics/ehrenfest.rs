use crate::initialization::Simulation;
use crate::interface::QuantumChemistryInterface;
use ::std::ops::AddAssign;
use ndarray::prelude::*;
use ndarray_linalg::c64;

impl Simulation {
    ///Ehrenfest dynamics routine of the struct Simulation
    pub fn ehrenfest_dynamics(&mut self, interface: &mut dyn QuantumChemistryInterface) {
        self.initialize_ehrenfest();

        for _step in 0..self.config.nstep {
            self.ehrenfest_step(interface);
        }
    }

    pub fn ehrenfest_step(&mut self, interface: &mut dyn QuantumChemistryInterface) {
        let old_forces: Array2<f64> = self.forces.clone();
        let old_energy: f64 = self.energies[self.state] + self.kinetic_energy;
        // calculate the gradient and the excitonic couplings
        let excitonic_couplings: Array2<f64> = self.get_ehrenfest_data(interface);
        // convert to complex array
        let excitonic_couplings: Array2<c64> =
            excitonic_couplings.map(|val| val * c64::new(1.0, 0.0));
        // surface hopping procedure
        // self.coefficients = self.ehrenfest_rk_integration(excitonic_couplings.view());
        self.coefficients = self.ehrenfest_sod_integration(excitonic_couplings.view());

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

    pub fn initialize_ehrenfest(&mut self) {
        // remove COM from coordinates
        self.coordinates = self.shift_to_center_of_mass();
        self.initial_coordinates = self.coordinates.clone();
        // remove tranlation and rotation
        self.velocities = self.eliminate_translation_rotation_from_velocity();
        // calculate the kinetic energy
        self.kinetic_energy = self.get_kinetic_energy();
        // Print output
        self.print_data(None);
        // Calculate new coordinates from velocity-verlet
        self.coordinates = self.get_coord_verlet();
        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();
    }

    pub fn get_ehrenfest_data(
        &mut self,
        interface: &mut dyn QuantumChemistryInterface,
    ) -> Array2<f64> {
        let abs_coefficients: Array1<f64> = self.coefficients.map(|val| val.norm());
        let tmp: (Array2<f64>, Array2<f64>) = interface.compute_ehrenfest(
            self.coordinates.view(),
            abs_coefficients.view(),
            self.config.ehrenfest_config.state_threshold,
        );

        self.energies = tmp.1.diag().to_owned();
        let forces: Array2<f64> = tmp.0;
        for (idx, mass) in self.masses.iter().enumerate() {
            self.forces
                .slice_mut(s![idx, ..])
                .assign(&(-1.0 * &forces.slice(s![idx, ..]) / mass.to_owned()));
        }

        // return the excitonic couplings
        tmp.1
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
