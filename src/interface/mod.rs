pub use ndarray::prelude::*;

/// Trait that provides an interface for a quantum chemistry programm.
/// The trait implements the function compute data,
/// which returns the energies, the gradient, the nonadiabatic couplings and the
/// dipoles of the molecular system
pub trait QuantumChemistryInterface {
    fn compute_data(
        &mut self,
        coordinates: ArrayView2<f64>,
        state: usize,
    ) -> (Array1<f64>, Array2<f64>, Array3<f64>, Array3<f64>);

    /// Returns the excitonic-coupling matrix and the gradient
    fn compute_ehrenfest(
        &mut self,
        coordinates: ArrayView2<f64>,
        state_coefficients: ArrayView1<f64>,
    ) -> (Array2<f64>, Array2<f64>);
}
