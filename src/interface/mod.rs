pub use ndarray::prelude::*;

pub trait QuantumChemistryInterface {
    fn compute_data(
        &mut self,
        coordinates: ArrayView2<f64>,
        state: usize,
    ) -> (Array1<f64>, Array2<f64>, Array3<f64>, Array3<f64>);
}
