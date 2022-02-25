use crate::constants;
use crate::dynamics::{align_nonadiabatic_coupling, get_nonadiabatic_scalar_coupling};
use crate::initialization::restart::read_restart_parameters;
use crate::initialization::Simulation;
use crate::interface::QuantumChemistryInterface;
use crate::output::*;
use ndarray::prelude::*;
use ndarray_linalg::c64;

impl Simulation {
    /// Initialize the velocity-verlet dynamic routine and print the first output of the dynamics simulation
    pub fn initialize_verlet(&mut self, interface: &mut dyn QuantumChemistryInterface) {
        if self.config.inputflag == "new" {
            self.initiate_trajectory(interface);
        } else if self.config.inputflag == "restart" {
            self.restart_trajectory(interface);
        } else {
            panic!("The inputflag must be either 'new' or 'restart'");
        }
        let restart: RestartOutput = RestartOutput::new(
            self.n_atoms,
            self.coordinates.view(),
            self.velocities.view(),
            self.nonadiabatic_arr.view(),
            self.coefficients.view(),
        );
        write_restart(&restart);

        let xyz_output: XyzOutput = XyzOutput::new(
            self.n_atoms,
            self.coordinates.view(),
            self.atomic_numbers.clone(),
        );
        write_xyz_custom(&xyz_output);

        let total_energy: f64 = self.kinetic_energy + self.energies[self.state];
        let energy_diff: f64 = total_energy;
        let full: StandardOutput = StandardOutput::new(
            0.0,
            self.coordinates.view(),
            self.velocities.view(),
            self.kinetic_energy,
            self.energies[self.state],
            total_energy,
            energy_diff,
            self.forces.view(),
            self.state,
        );
        write_full_custom(&full, self.masses.view());
        write_state(self.state);

        // Calculate new coordinates from velocity-verlet
        self.coordinates = self.get_coord_verlet();
        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();
    }

    /// Calculate a single step of the velocity-verlet dynamics utilizing the [QuantumChemistryInterface]
    /// for the calculation of the required properties
    pub fn verlet_step(&mut self, interface: &mut dyn QuantumChemistryInterface) {
        let old_forces: Array2<f64> = self.forces.clone();
        let old_energy: f64 = self.energies[self.state] + self.kinetic_energy;
        let old_kinetic: f64 = self.kinetic_energy;
        let old_potential_energy: f64 = self.energies[self.state];
        let last_energies: Array1<f64> = self.energies.clone();
        let old_state: usize = self.state;
        let econst: f64 = self.config.hopping_config.start_econst * constants::FS_TO_AU;

        // calculate energies, forces, dipoles, nonadiabatic_scalar
        // for the new geometry
        self.get_quantum_chem_data(interface);

        if self.config.hopping_config.coupling_flag > -1 {
            if self.config.gs_dynamic {
                // skip hopping procedure if the ground state is forced
            } else {
                let old_coeff: Array1<c64> = self.coefficients.clone();
                // integration of the schroedinger equation
                // employing a runge-kutta scheme
                if self.config.hopping_config.integration_type == "RK" {
                    self.coefficients = self.new_coefficients();
                } else if self.config.hopping_config.integration_type == "LD" {
                    let tmp: (Array1<c64>, Array2<f64>, Array2<f64>) =
                        self.get_local_diabatization(last_energies.view(), self.t_tot_last.clone());
                    self.coefficients = tmp.0;
                    self.t_tot_last = Some(tmp.2);
                } else {
                    // automatic choice of integration method
                    if self.config.hopping_config.coupling_flag == 1 {
                        // in the absence of an external field (jflag == 1) the
                        // coefficients are integrated in the local diabatic basis.
                        // The diabatization procedure requires the overlap matrix
                        // between wavefunctions at different time steps.
                        let tmp: (Array1<c64>, Array2<f64>, Array2<f64>) = self
                            .get_local_diabatization(last_energies.view(), self.t_tot_last.clone());
                        self.coefficients = tmp.0;
                        self.t_tot_last = Some(tmp.2);
                    } else {
                        self.coefficients = self.new_coefficients();
                    }
                }
                // calculate the state of the simulation after the hopping procedure
                self.get_new_state(old_coeff.view());

                if self.actual_time > econst && self.config.hopping_config.decoherence_correction {
                    // The decoherence correction should be turned on only
                    // if energy conservation is turned on, too.
                    // During the action of an explicit electric field pulse,
                    // the decoherence correction should be turned off.
                    self.coefficients = self.get_decoherence_correction(0.1);
                }
            }
            // Rescale the velocities after a population transfer
            if self.actual_time > econst && self.state != old_state {
                if self.config.hopping_config.rescale_type == "uniform" {
                    let tmp: (Array2<f64>, usize) = self.uniformly_rescaled_velocities(old_state);
                    self.state = tmp.1;
                    self.velocities = tmp.0;
                } else if self.config.hopping_config.rescale_type == "vector" {
                    let tmp: (Array2<f64>, usize) =
                        self.rescaled_velocities(old_state, self.config.nstates - 1);
                    self.state = tmp.1;
                    self.velocities = tmp.0;
                }
            }
        }
        self.actual_step += self.config.hopping_config.integration_steps as f64;

        // Calculate new coordinates from velocity-verlet
        self.velocities = self.get_velocities_verlet(old_forces.view());
        // remove tranlation and rotation from the velocities
        self.velocities = self.eliminate_translation_rotation_from_velocity();

        // calculate the kinetic energy
        self.kinetic_energy = self.get_kinetic_energy();

        // scale velocities
        if self.config.dyn_mode == 'T' {
            self.velocities = self
                .thermostat
                .scale_velocities(self.velocities.view(), self.kinetic_energy);
            // (self.velocities.view(),self.kinetic_energy);            self.velocities = self.scale_velocities_temperature();
        }
        if self.config.dyn_mode == 'E' && self.config.artificial_energy_conservation {
            self.velocities =
                self.scale_velocities_const_energy(old_state, old_kinetic, old_potential_energy);
        }
        // Write Output in each step
        let restart: RestartOutput = RestartOutput::new(
            self.n_atoms,
            self.coordinates.view(),
            self.velocities.view(),
            self.nonadiabatic_arr.view(),
            self.coefficients.view(),
        );
        write_restart(&restart);

        let xyz_output: XyzOutput = XyzOutput::new(
            self.n_atoms,
            self.coordinates.view(),
            self.atomic_numbers.clone(),
        );
        write_xyz_custom(&xyz_output);

        let total_energy: f64 = self.kinetic_energy + self.energies[self.state];
        let energy_diff: f64 = total_energy - old_energy;

        let full: StandardOutput = StandardOutput::new(
            self.actual_time,
            self.coordinates.view(),
            self.velocities.view(),
            self.kinetic_energy,
            self.energies[self.state],
            total_energy,
            energy_diff,
            self.forces.view(),
            self.state,
        );
        write_full_custom(&full, self.masses.view());
        write_energies(self.energies.view());
        write_state(self.state);

        if self.config.hopping_config.coupling_flag > -1 {
            let hopping_out: HoppingOutput = HoppingOutput::new(
                self.actual_time,
                self.coefficients.view(),
                self.nonadiabatic_scalar.view(),
                self.dipole.view(),
            );
            write_hopping(&hopping_out);
        }

        // Calculate new coordinates from velocity-verlet
        self.coordinates = self.get_coord_verlet();

        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();

        // update the actual time
        self.actual_time += self.stepsize;
    }

    /// Initialize the langevin dynamics and write the first output of the simulation
    pub fn initialize_langevin(&mut self, interface: &mut dyn QuantumChemistryInterface) {
        if self.config.inputflag == "new" {
            self.initiate_trajectory(interface);
        } else if self.config.inputflag == "restart" {
            self.restart_trajectory(interface);
        } else {
            panic!("The inputflag must be either 'new' or 'restart'");
        }

        // Write initial output
        let restart: RestartOutput = RestartOutput::new(
            self.n_atoms,
            self.coordinates.view(),
            self.velocities.view(),
            self.nonadiabatic_arr.view(),
            self.coefficients.view(),
        );
        write_restart(&restart);
        // write_restart_custom(&restart);

        let xyz_output: XyzOutput = XyzOutput::new(
            self.n_atoms,
            self.coordinates.view(),
            self.atomic_numbers.clone(),
        );
        write_xyz_custom(&xyz_output);

        let total_energy: f64 = self.kinetic_energy + self.energies[self.state];
        let energy_diff: f64 = total_energy;
        let full: StandardOutput = StandardOutput::new(
            0.0,
            self.coordinates.view(),
            self.velocities.view(),
            self.kinetic_energy,
            self.energies[self.state],
            total_energy,
            energy_diff,
            self.forces.view(),
            self.state,
        );
        write_full_custom(&full, self.masses.view());
        write_state(self.state);
        write_energies(self.energies.view());

        // Langevin routine
        let (_vrand, prand): (Array2<f64>, Array2<f64>) = self.get_random_terms();
        self.saved_p_rand = prand;
        let efactor = self.get_e_factor_langevin();
        self.saved_efactor = efactor;

        // get new coordinates
        self.coordinates = self.get_coordinates_langevin();
        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();
    }

    /// Calculate a single step of the langevin dynamics
    pub fn langevin_step(&mut self, interface: &mut dyn QuantumChemistryInterface) {
        let econst: f64 = self.config.hopping_config.start_econst * constants::FS_TO_AU;
        let old_forces: Array2<f64> = self.forces.clone();
        let old_energy: f64 = self.energies[self.state] + self.kinetic_energy;
        let last_energies: Array1<f64> = self.energies.clone();
        let old_state: usize = self.state;

        // calculate energies, forces, dipoles, nonadiabatic_scalar
        // for the new geometry
        self.get_quantum_chem_data(interface);

        if self.config.hopping_config.coupling_flag > -1 {
            if self.config.gs_dynamic {
                // skip hopping procedure if the ground state is forced
            } else {
                let old_coeff: Array1<c64> = self.coefficients.clone();
                // integration of the schroedinger equation
                // employing a runge-kutta scheme
                if self.config.hopping_config.integration_type == "RK" {
                    self.coefficients = self.new_coefficients();
                } else if self.config.hopping_config.integration_type == "LD" {
                    let tmp: (Array1<c64>, Array2<f64>, Array2<f64>) =
                        self.get_local_diabatization(last_energies.view(), self.t_tot_last.clone());
                    self.coefficients = tmp.0;
                    self.t_tot_last = Some(tmp.2);
                } else {
                    // automatic choice of integration method
                    if self.config.hopping_config.coupling_flag == 1 {
                        // in the absence of an external field (jflag == 1) the
                        // coefficients are integrated in the local diabatic basis.
                        // The diabatization procedure requires the overlap matrix
                        // between wavefunctions at different time steps.
                        // let s_mat: Array2<f64> = self.s_mat.clone().unwrap();
                        let tmp: (Array1<c64>, Array2<f64>, Array2<f64>) = self
                            .get_local_diabatization(last_energies.view(), self.t_tot_last.clone());
                        self.coefficients = tmp.0;
                        self.t_tot_last = Some(tmp.2);
                    } else {
                        self.coefficients = self.new_coefficients();
                    }
                }
                // calculate the state of the simulation after the hopping procedure
                self.get_new_state(old_coeff.view());

                if self.actual_time > econst && self.config.hopping_config.decoherence_correction {
                    // The decoherence correction should be turned on only
                    // if energy conservation is turned on, too.
                    // During the action of an explicit electric field pulse,
                    // the decoherence correction should be turned off.
                    self.coefficients = self.get_decoherence_correction(0.1);
                }
            }
            // Rescale the velocities after a population transfer
            if self.actual_time > econst && self.state != old_state {
                if self.config.hopping_config.rescale_type == "uniform" {
                    let tmp: (Array2<f64>, usize) = self.uniformly_rescaled_velocities(self.state);
                    self.state = tmp.1;
                    self.velocities = tmp.0;
                } else if self.config.hopping_config.rescale_type == "vector" {
                    let tmp: (Array2<f64>, usize) =
                        self.rescaled_velocities(old_state, self.config.nstates - 1);
                    self.state = tmp.1;
                    self.velocities = tmp.0;
                }
            }
        }
        self.actual_step += self.config.hopping_config.integration_steps as f64;

        let (vrand, prand): (Array2<f64>, Array2<f64>) = self.get_random_terms();
        self.saved_p_rand = prand;
        let efactor = self.get_e_factor_langevin();
        self.saved_efactor = efactor;
        // calculate new velocities
        self.velocities = self.get_velocities_langevin(old_forces.view(), vrand.view());
        // remove translation and rotation from velocities
        self.velocities = self.eliminate_translation_rotation_from_velocity();
        // calculate kinetic energy
        self.kinetic_energy = self.get_kinetic_energy();

        // Write Output in each step
        let restart: RestartOutput = RestartOutput::new(
            self.n_atoms,
            self.coordinates.view(),
            self.velocities.view(),
            self.nonadiabatic_arr.view(),
            self.coefficients.view(),
        );
        write_restart(&restart);
        // write_restart_custom(&restart);

        let xyz_output: XyzOutput = XyzOutput::new(
            self.n_atoms,
            self.coordinates.view(),
            self.atomic_numbers.clone(),
        );
        write_xyz_custom(&xyz_output);

        let total_energy: f64 = self.kinetic_energy + self.energies[self.state];
        let energy_diff: f64 = total_energy - old_energy;
        let full: StandardOutput = StandardOutput::new(
            self.actual_time,
            self.coordinates.view(),
            self.velocities.view(),
            self.kinetic_energy,
            self.energies[self.state],
            total_energy,
            energy_diff,
            self.forces.view(),
            self.state,
        );
        write_full_custom(&full, self.masses.view());
        write_energies(self.energies.view());
        write_state(self.state);

        if self.config.hopping_config.coupling_flag > -1 {
            let hopping_out: HoppingOutput = HoppingOutput::new(
                self.actual_time,
                self.coefficients.view(),
                self.nonadiabatic_scalar.view(),
                self.dipole.view(),
            );
            write_hopping(&hopping_out);
        }

        // calculate new coordinates
        self.coordinates = self.get_coordinates_langevin();

        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();

        self.actual_time += self.stepsize;
    }

    /// Do the first calculation of the energies, gradient, nonadiabatic couplings and the dipoles
    /// using the [QuantumChemistryInterface]
    pub fn initialize_quantum_chem_interface(
        &mut self,
        interface: &mut dyn QuantumChemistryInterface,
    ) {
        let tmp: (Array1<f64>, Array2<f64>, Array3<f64>, Array3<f64>) =
            interface.compute_data(self.coordinates.view(), self.state);

        self.energies = tmp.0;
        let forces: Array2<f64> = tmp.1;
        for (idx, mass) in self.masses.iter().enumerate() {
            self.forces
                .slice_mut(s![idx, ..])
                .assign(&(-1.0 * &forces.slice(s![idx, ..]) / mass.to_owned()));
        }

        // self.forces = tmp.1.mapv(|val| (val * 1.0e9).round() / 1.0e9);
        self.nonadiabatic_arr = tmp.2.clone();
        self.nonadiabatic_arr_old = tmp.2;
        self.dipole = tmp.3;

        //calculate nonadiabatic scalar coupling
        if self.config.hopping_config.coupling_flag == 1
            || self.config.hopping_config.coupling_flag == 2
        {
            self.nonadiabatic_scalar = get_nonadiabatic_scalar_coupling(
                self.config.nstates,
                0,
                self.config.nstates - 1,
                self.nonadiabatic_arr.view(),
                self.velocities.view(),
            );
        }

        self.nonadiabatic_scalar_old = self.nonadiabatic_scalar.clone();
        self.s_mat = self.nonadiabatic_scalar.clone() * self.stepsize;
    }

    /// Calculate the energies, gradient, nonadiabatic couplings and the dipoles
    /// using the [QuantumChemistryInterface]
    pub fn get_quantum_chem_data(&mut self, interface: &mut dyn QuantumChemistryInterface) {
        // calculate energy, forces, etc for new coords
        // let mut handler: Bagel_Handler = self.handler.clone().unwrap();

        let tmp: (Array1<f64>, Array2<f64>, Array3<f64>, Array3<f64>) =
            interface.compute_data(self.coordinates.view(), self.state);

        self.energies = tmp.0;
        let forces: Array2<f64> = tmp.1;
        for (idx, mass) in self.masses.iter().enumerate() {
            self.forces
                .slice_mut(s![idx, ..])
                .assign(&(-1.0 * &forces.slice(s![idx, ..]) / mass.to_owned()));
        }
        // self.forces = tmp.1.mapv(|val| (val * 1.0e9).round() / 1.0e9);
        self.nonadiabatic_arr = tmp.2;
        self.nonadiabatic_arr = align_nonadiabatic_coupling(
            self.nonadiabatic_arr_old.view(),
            self.nonadiabatic_arr.view(),
            self.n_atoms,
        );
        self.dipole_old = self.dipole.clone();
        self.dipole = tmp.3;
        self.nonadiabatic_arr_old = self.nonadiabatic_arr.clone();

        //calculate nonadiabatic scalar coupling
        if self.config.hopping_config.coupling_flag == 1
            || self.config.hopping_config.coupling_flag == 2
        {
            self.nonadiabatic_scalar = get_nonadiabatic_scalar_coupling(
                self.config.nstates,
                0,
                self.config.nstates - 1,
                self.nonadiabatic_arr.view(),
                self.velocities.view(),
            );
        }
        self.nonadiabatic_scalar_old = self.nonadiabatic_scalar.clone();
        self.s_mat = self.nonadiabatic_scalar.clone() * self.stepsize;
    }

    /// Initiate the trajectory
    pub fn initiate_trajectory(&mut self, interface: &mut dyn QuantumChemistryInterface) {
        self.coordinates = self.shift_to_center_of_mass();
        self.velocities = self.eliminate_translation_rotation_from_velocity();

        self.initialize_quantum_chem_interface(interface);

        self.kinetic_energy = self.get_kinetic_energy();
    }

    /// Restart the trajectory
    pub fn restart_trajectory(&mut self, interface: &mut dyn QuantumChemistryInterface) {
        let temp: (Array2<f64>, Array2<f64>, Array3<f64>, Array1<c64>) = read_restart_parameters();
        self.coordinates = temp.0;
        self.velocities = temp.1;
        self.nonadiabatic_arr_old = temp.2;
        self.coefficients = temp.3;

        // calculate quantum chemical data
        self.get_quantum_chem_data(interface);
        self.kinetic_energy = self.get_kinetic_energy();
    }
}
