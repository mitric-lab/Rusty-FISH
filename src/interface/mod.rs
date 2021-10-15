pub mod bagel;
pub use bagel::*;
pub use ndarray::prelude::*;

pub trait QuantumChemistryInterface{
    fn compute_data(&mut self,coordinates:ArrayView2<f64>,state:usize)
        ->(Array1<f64>, Array2<f64>, Array3<f64>, Array3<f64>);
}

impl QuantumChemistryInterface for Bagel_Handler{
    fn compute_data(&mut self, coordinates: ArrayView2<f64>, state: usize) -> (Array1<f64>, Array2<f64>, Array3<f64>, Array3<f64>) {
        self.get_all(coordinates,state)
    }
}