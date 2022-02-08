use ndarray::{Array, Array2, ArrayView2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::activation::Sigmoid;

pub struct Linear {
    inputs: Array2<f64>,
    weights: Array2<f64>,
    dot_products: Array2<f64>,
    outputs: Array2<f64>,
    activation: Sigmoid,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize, activation: Sigmoid) -> Self {
        let inputs = Array2::zeros((input_dim, 1));
        let weights = Array::random((output_dim, input_dim), Uniform::new(0.0, 1.0));
        let dot_products = Array2::zeros((output_dim, 1));
        let outputs = Array2::zeros((output_dim, 1));
        Self {
            inputs,
            weights,
            dot_products,
            outputs,
            activation,
        }
    }

    pub fn forward(&mut self, inputs: Array2<f64>) -> ArrayView2<f64> {
        self.inputs = inputs;
        self.dot_products = self.weights.dot(&self.inputs);
        self.outputs = self.activation.compute(&self.dot_products);
        self.outputs.view()
    }

    pub fn backward(&mut self, errors: &ArrayView2<f64>) -> (Array2<f64>, Array2<f64>) {
        let activation_derivative = self.activation.derivative(&self.dot_products);
        let dot_products_derivative = activation_derivative * errors;
        let inputs_derivative = self.weights.t().dot(&dot_products_derivative);
        let weights_derivative = dot_products_derivative.dot(&self.inputs);
        (inputs_derivative, weights_derivative)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn layer_forward() {
        let mut layer = Linear::new(3, 2, Sigmoid);
        let inputs = arr2(&[[1.0], [0.5], [-0.5]]);
        let outputs = layer.forward(inputs);
        let train = arr2(&[[1.0, 0.0, 0.0]]);
        let error = train - outputs;
        layer.backward(&error.view());
    }
}
