use crate::dynamics::{
    align_nonadiabatic_coupling, extrapolate_forces, get_nonadiabatic_scalar_coupling,
};
use crate::initialization::restart;
use crate::initialization::Simulation;
use crate::interface::bagel::*;
use crate::output::*;
use ndarray::prelude::*;
use ndarray_linalg::c64;
use crate::initialization::restart::read_restart_parameters;

impl Simulation<'_> {
    pub fn verlet_dynamics(
        &mut self,
    ) {
        if self.config.inputflag == "new" {
            self.initiate_trajectory();
        } else if self.config.inputflag == "restart" {
            self.restart_trajectory();
        }
        else{
            panic!("The inputflag must be either 'new' or 'restart'");
        }
        let mut old_coords: Array2<f64> = self.coordinates.clone();

        if self.config.extrapolate_forces == true {
            for i in (0..3) {
                self.last_forces
                    .slice_mut(s![i, .., ..])
                    .assign(&self.forces);
            }
        }
        // Write initial output
        let restart: Restart_Output = Restart_Output::new(
            self.n_atoms,
            self.coordinates.view(),
            self.velocities.view(),
            self.nonadiabatic_arr.view(),
            self.coefficients.view(),
        );
        write_restart(&restart);
        // write_restart_custom(&restart);

        let xyz_output: XYZ_Output = XYZ_Output::new(
            self.n_atoms,
            self.coordinates.view(),
            self.atomic_numbers.clone(),
        );
        write_xyz_custom(&xyz_output);

        let total_energy: f64 = self.kinetic_energy + self.energies[self.state];
        let energy_diff: f64 = total_energy;
        let full: Standard_Output = Standard_Output::new(
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

        // Calculate new coordinates from velocity-verlet
        self.coordinates = self.get_coord_verlet();
        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();

        let mut time: f64 = self.stepsize;
        let mut actual_step: f64 = 0.0;
        let mut t_tot_last: Option<Array2<f64>> = None;

        // start of the actual dynamics
        for step in (1..self.config.nstep + 1) {
            // save parameters of the previous iteration
            let old_forces: Array2<f64> = self.forces.clone();
            let old_energy: f64 = self.energies[self.state] + self.kinetic_energy;
            let old_kinetic: f64 = self.kinetic_energy;
            let old_potential_energy: f64 = self.energies[self.state];
            let last_energies: Array1<f64> = self.energies.clone();
            let old_state: usize = self.state;

            // initilization of the force extrapolation
            if self.config.extrapolate_forces == true {
                let index: usize = (step - 1).rem_euclid(3);
                self.last_forces
                    .slice_mut(s![index, .., ..])
                    .assign(&self.forces);
            }

            // calculate energies, forces, dipoles, nonadiabatic_scalar
            // for the new geometry
            self.get_quantum_chem_data();

            if self.config.extrapolate_forces == true {
                let index: usize = (step - 1).rem_euclid(3);
                // extrapolate forces
                self.forces =
                    extrapolate_forces(self.last_forces.view(), index, self.forces.view());
            }

            if self.config.coupling > -1 {
                if self.config.gs_dynamic == true {
                    // skip hopping procedure if the ground state is forced
                    continue;
                } else {
                    let old_coeff: Array1<c64> = self.coefficients.clone();
                    // integration of the schroedinger equation
                    // employing a runge-kutta scheme
                    if self.config.integration_type == "RK" {
                        self.coefficients = self.get_hopping_fortran(actual_step);
                    } else if self.config.integration_type == "LD" {
                        let tmp: (Array1<c64>, Array2<f64>, Array2<f64>) =
                            self.get_local_diabatization(last_energies.view(), t_tot_last.clone());
                        self.coefficients = tmp.0;
                        t_tot_last = Some(tmp.2);
                    } else {
                        // automatic choice of integration method
                        if self.config.coupling == 1 {
                            // in the absence of an external field (jflag == 1) the
                            // coefficients are integrated in the local diabatic basis.
                            // The diabatization procedure requires the overlap matrix
                            // between wavefunctions at different time steps.
                            // let s_mat: Array2<f64> = self.s_mat.clone().unwrap();
                            let tmp: (Array1<c64>, Array2<f64>, Array2<f64>) = self
                                .get_local_diabatization(last_energies.view(), t_tot_last.clone());
                            self.coefficients = tmp.0;
                            t_tot_last = Some(tmp.2);
                        } else {
                            self.coefficients = self.get_hopping_fortran(actual_step);
                        }
                    }
                    // calculate the state of the simulation after the hopping procedure
                    self.get_new_state(old_coeff.view());

                    if time > self.start_econst && self.config.decoherence_correction == true {
                        // The decoherence correction should be turned on only
                        // if energy conservation is turned on, too.
                        // During the action of an explicit electric field pulse,
                        // the decoherence correction should be turned off.
                        self.coefficients = self.get_decoherence_correction(0.1);
                    }
                }
                // Rescale the velocities after a population transfer
                if time > self.start_econst && self.state != old_state {
                    if self.config.rescale_type == "uniform" {
                        let tmp: (Array2<f64>, usize) =
                            self.uniformly_rescaled_velocities(old_state);
                        self.state = tmp.1;
                        self.velocities = tmp.0;
                    } else if self.config.rescale_type == "vector" {
                        let tmp: (Array2<f64>, usize) =
                            self.rescaled_velocities(old_state, self.config.nstates - 1);
                        self.state = tmp.1;
                        self.velocities = tmp.0;
                    }
                }
            }
            actual_step += self.config.n_small_steps as f64;

            // Calculate new coordinates from velocity-verlet
            self.velocities = self.get_velocities_verlet(old_forces.view());
            // remove tranlation and rotation from the velocities
            self.velocities = self.eliminate_translation_rotation_from_velocity();

            // calculate the kinetic energy
            self.kinetic_energy = self.get_kinetic_energy();

            // scale velocities
            if self.config.dyn_mode == 'T' {
                self.velocities = self.scale_velocities_temperature();
            }
            if self.config.dyn_mode == 'E' && self.config.artificial_energy_conservation == true {
                self.velocities = self.scale_velocities_const_energy(
                    old_state,
                    old_kinetic,
                    old_potential_energy,
                );
            }
            // Write Output in each step
            let restart: Restart_Output = Restart_Output::new(
                self.n_atoms,
                self.coordinates.view(),
                self.velocities.view(),
                self.nonadiabatic_arr.view(),
                self.coefficients.view(),
            );
            write_restart(&restart);
            // write_restart_custom(&restart);

            let xyz_output: XYZ_Output = XYZ_Output::new(
                self.n_atoms,
                self.coordinates.view(),
                self.atomic_numbers.clone(),
            );
            write_xyz_custom(&xyz_output);

            let total_energy: f64 = self.kinetic_energy + self.energies[self.state];
            let energy_diff: f64 = total_energy - old_energy;

            let full: Standard_Output = Standard_Output::new(
                time,
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

            if self.config.coupling > -1 {
                let hopping_out: hopping_output = hopping_output::new(
                    time,
                    self.coefficients.view(),
                    self.nonadiabatic_scalar.view(),
                    self.dipole.view(),
                );
                write_hopping(&hopping_out);
            }

            if step < self.config.nstep {
                old_coords = self.coordinates.clone();

                // Calculate new coordinates from velocity-verlet
                self.coordinates = self.get_coord_verlet();

                // Shift coordinates to center of mass
                self.coordinates = self.shift_to_center_of_mass();

                time += self.stepsize;
            }
        }
    }

    pub fn langevin_dynamics(&mut self) {
        if self.config.inputflag == "new" {
            self.initiate_trajectory();
        } else if self.config.inputflag == "restart" {
            self.restart_trajectory();
        }
        else{
            panic!("The inputflag must be either 'new' or 'restart'");
        }
        let mut old_coords: Array2<f64> = self.coordinates.clone();

        if self.config.extrapolate_forces == true {
            for i in (0..3) {
                self.last_forces
                    .slice_mut(s![i, .., ..])
                    .assign(&self.forces);
            }
        }
        // Write initial output
        let restart: Restart_Output = Restart_Output::new(
            self.n_atoms,
            self.coordinates.view(),
            self.velocities.view(),
            self.nonadiabatic_arr.view(),
            self.coefficients.view(),
        );
        write_restart(&restart);
        // write_restart_custom(&restart);

        let xyz_output: XYZ_Output = XYZ_Output::new(
            self.n_atoms,
            self.coordinates.view(),
            self.atomic_numbers.clone(),
        );
        write_xyz_custom(&xyz_output);

        let total_energy: f64 = self.kinetic_energy + self.energies[self.state];
        let energy_diff: f64 = total_energy;
        let full: Standard_Output = Standard_Output::new(
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
        let (vrand, prand): (Array2<f64>, Array2<f64>) = self.get_random_terms();
        self.saved_p_rand = prand;
        let efactor = self.get_e_factor_langevin();
        self.saved_efactor = efactor;

        // get new coordinates
        self.coordinates = self.get_coordinates_langevin();
        // Shift coordinates to center of mass
        self.coordinates = self.shift_to_center_of_mass();

        let mut time: f64 = self.stepsize;
        let mut actual_step: f64 = 0.0;
        let mut t_tot_last: Option<Array2<f64>> = None;

        // start of the actual dynamics
        for step in (1..self.config.nstep + 1) {
            // save parameters of the previous iteration
            let old_forces: Array2<f64> = self.forces.clone();
            let old_energy: f64 = self.energies[self.state] + self.kinetic_energy;
            let last_energies: Array1<f64> = self.energies.clone();
            let old_state: usize = self.state;

            // initilization of the force extrapolation
            if self.config.extrapolate_forces == true {
                let index: usize = (step - 1).rem_euclid(3);
                self.last_forces
                    .slice_mut(s![index, .., ..])
                    .assign(&self.forces);
            }

            // calculate energies, forces, dipoles, nonadiabatic_scalar
            // for the new geometry
            self.get_quantum_chem_data();

            if self.config.extrapolate_forces == true {
                let index: usize = (step - 1).rem_euclid(3);
                // extrapolate forces
                self.forces =
                    extrapolate_forces(self.last_forces.view(), index, self.forces.view());
            }

            if self.config.coupling > -1 {
                if self.config.gs_dynamic == true {
                    // skip hopping procedure if the ground state is forced
                    continue;
                } else {
                    let old_coeff: Array1<c64> = self.coefficients.clone();
                    // integration of the schroedinger equation
                    // employing a runge-kutta scheme
                    if self.config.integration_type == "RK" {
                        self.coefficients = self.get_hopping_fortran(actual_step);
                    } else if self.config.integration_type == "LD" {
                        let tmp: (Array1<c64>, Array2<f64>, Array2<f64>) =
                            self.get_local_diabatization(last_energies.view(), t_tot_last.clone());
                        self.coefficients = tmp.0;
                        t_tot_last = Some(tmp.2);
                    } else {
                        // automatic choice of integration method
                        if self.config.coupling == 1 {
                            // in the absence of an external field (jflag == 1) the
                            // coefficients are integrated in the local diabatic basis.
                            // The diabatization procedure requires the overlap matrix
                            // between wavefunctions at different time steps.
                            // let s_mat: Array2<f64> = self.s_mat.clone().unwrap();
                            let tmp: (Array1<c64>, Array2<f64>, Array2<f64>) = self
                                .get_local_diabatization(last_energies.view(), t_tot_last.clone());
                            self.coefficients = tmp.0;
                            t_tot_last = Some(tmp.2);
                        } else {
                            self.coefficients = self.get_hopping_fortran(actual_step);
                        }
                    }
                    // calculate the state of the simulation after the hopping procedure
                    self.get_new_state(old_coeff.view());

                    if time > self.start_econst && self.config.decoherence_correction == true {
                        // The decoherence correction should be turned on only
                        // if energy conservation is turned on, too.
                        // During the action of an explicit electric field pulse,
                        // the decoherence correction should be turned off.
                        self.coefficients = self.get_decoherence_correction(0.1);
                    }
                }
                // Rescale the velocities after a population transfer
                if time > self.start_econst && self.state != old_state {
                    if self.config.rescale_type == "uniform" {
                        let tmp: (Array2<f64>, usize) =
                            self.uniformly_rescaled_velocities(self.state);
                        self.state = tmp.1;
                        self.velocities = tmp.0;
                    } else if self.config.rescale_type == "vector" {
                        let tmp: (Array2<f64>, usize) =
                            self.rescaled_velocities(old_state, self.config.nstates - 1);
                        self.state = tmp.1;
                        self.velocities = tmp.0;
                    }
                }
            }
            actual_step += self.config.n_small_steps as f64;

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
            let restart: Restart_Output = Restart_Output::new(
                self.n_atoms,
                self.coordinates.view(),
                self.velocities.view(),
                self.nonadiabatic_arr.view(),
                self.coefficients.view(),
            );
            write_restart(&restart);
            // write_restart_custom(&restart);

            let xyz_output: XYZ_Output = XYZ_Output::new(
                self.n_atoms,
                self.coordinates.view(),
                self.atomic_numbers.clone(),
            );
            write_xyz_custom(&xyz_output);

            let total_energy: f64 = self.kinetic_energy + self.energies[self.state];
            let energy_diff: f64 = total_energy - old_energy;
            let full: Standard_Output = Standard_Output::new(
                time,
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

            if self.config.coupling > -1 {
                let hopping_out: hopping_output = hopping_output::new(
                    time,
                    self.coefficients.view(),
                    self.nonadiabatic_scalar.view(),
                    self.dipole.view(),
                );
                write_hopping(&hopping_out);
            }

            if step < self.config.nstep {
                old_coords = self.coordinates.clone();

                // calculate new coordinates
                self.coordinates = self.get_coordinates_langevin();

                // Shift coordinates to center of mass
                self.coordinates = self.shift_to_center_of_mass();

                time += self.stepsize;
            }
        }
    }

    pub fn initialize_quantum_chem_interface(&mut self,) {
        let tmp: (Array1<f64>, Array2<f64>, Array3<f64>, Array3<f64>) =
            self.interface.compute_data(self.coordinates.view(), self.state);

        self.energies = tmp.0;
        let mut forces:Array2<f64> = tmp.1;
        for (idx,mass) in self.masses.iter().enumerate(){
            self.forces.slice_mut(s![idx,..]).assign(&(-1.0 * &forces.slice(s![idx,..])/mass.to_owned()));
        }

        // self.forces = tmp.1.mapv(|val| (val * 1.0e9).round() / 1.0e9);
        self.nonadiabatic_arr = tmp.2.clone();
        self.nonadiabatic_arr_old = tmp.2;
        self.dipole = tmp.3;

        //calculate nonadiabatic scalar coupling
        if self.config.coupling == 1 || self.config.coupling == 2 {
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

    pub fn get_quantum_chem_data(&mut self) {
        // calculate energy, forces, etc for new coords
        // let mut handler: Bagel_Handler = self.handler.clone().unwrap();

        let tmp: (Array1<f64>, Array2<f64>, Array3<f64>, Array3<f64>) =
            self.interface.compute_data(self.coordinates.view(), self.state);

        self.energies = tmp.0;
        let mut forces:Array2<f64> = tmp.1;
        for (idx,mass) in self.masses.iter().enumerate(){
            self.forces.slice_mut(s![idx,..]).assign(&(-1.0 * &forces.slice(s![idx,..])/mass.to_owned()));
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
        if self.config.coupling == 1 || self.config.coupling == 2 {
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

    pub fn initiate_trajectory(&mut self) {
        self.coordinates = self.shift_to_center_of_mass();
        self.velocities = self.eliminate_translation_rotation_from_velocity();

        self.initialize_quantum_chem_interface();

        self.kinetic_energy = self.get_kinetic_energy();
    }

    pub fn restart_trajectory(&mut self){
        let temp:(Array2<f64>,Array2<f64>,Array3<f64>,Array1<c64>) = read_restart_parameters();
        self.coordinates = temp.0;
        self.velocities = temp.1;
        self.nonadiabatic_arr_old = temp.2;
        self.coefficients = temp.3;

        self.coordinates = self.shift_to_center_of_mass();
        self.velocities = self.eliminate_translation_rotation_from_velocity();

        // calculate quantum chemical data
        self.get_quantum_chem_data();

        self.kinetic_energy = self.get_kinetic_energy();
    }
}
