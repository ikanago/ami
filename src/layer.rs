use ndarray::{Array, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::activation::Activation;

pub struct Linear {
    inputs: Array2<f64>,
    weights: Array2<f64>,
    dot_products: Array2<f64>,
    outputs: Array2<f64>,
    activation: Box<dyn Activation>,
}

impl Linear {
    pub fn new<A: Activation + 'static>(
        input_dim: usize,
        output_dim: usize,
        batch_size: usize,
        activation: A,
    ) -> Self {
        let weights = Array::random((input_dim + 1, output_dim), Uniform::new(0.0, 1.0));
        Linear::with_weights(input_dim, output_dim, batch_size, activation, weights)
    }

    pub fn with_weights<A: Activation + 'static>(
        input_dim: usize,
        output_dim: usize,
        batch_size: usize,
        activation: A,
        weights: Array2<f64>,
    ) -> Self {
        let inputs = Array2::zeros((batch_size, input_dim + 1));
        let dot_products = Array2::zeros((batch_size, output_dim));
        let outputs = Array2::zeros((batch_size, output_dim));
        Self {
            inputs,
            weights,
            dot_products,
            outputs,
            activation: Box::new(activation),
        }
    }

    pub fn forward(&mut self, inputs: Array2<f64>) -> Array2<f64> {
        let batch_size = inputs.nrows();
        let inputs_for_bias = Array::<f64, _>::ones((batch_size, 1));
        let mut inputs = inputs;
        inputs.append(Axis(1), inputs_for_bias.view()).unwrap();

        self.inputs = inputs;
        self.dot_products = self.inputs.dot(&self.weights);
        self.outputs = self.activation.compute(&self.dot_products);
        self.outputs.to_owned()
    }

    pub fn backward(&mut self, errors: Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let activation_derivative = self.activation.derivative(&self.dot_products);
        let dot_products_derivative = activation_derivative * errors;
        let mut inputs_derivative = dot_products_derivative.dot(&self.weights.t());
        let weights_derivative = self.inputs.t().dot(&dot_products_derivative);

        inputs_derivative.remove_index(Axis(1), inputs_derivative.ncols() - 1);
        (inputs_derivative, weights_derivative)
    }

    pub fn update_weights(&mut self, learning_rate: f64, mut weights_derivative: Array2<f64>) {
        weights_derivative *= learning_rate;
        self.weights -= &weights_derivative;
    }
}

#[cfg(test)]
mod tests {
    use crate::{activation::Sigmoid, assert_rel_eq_arr2};

    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn layer_forward() {
        let weights = arr2(&[[1.0, 2.0], [-1.0, -1.0], [0.5, 2.0], [-2.0, -0.5]]);
        let mut layer = Linear::with_weights(3, 2, 2, Sigmoid, weights);
        let inputs = arr2(&[[1.0, 0.5, -0.5], [0.0, 1.0, 0.5]]);
        let outputs = layer.forward(inputs);
        assert_rel_eq_arr2!(
            outputs,
            arr2(&[
                [0.1480471980316895, 0.5],
                [0.0600866501740076, 0.3775406687981454],
            ])
        );

        let train = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let error = train - outputs;
        let (dz, dw) = layer.backward(error);

        assert_rel_eq_arr2!(
            dz,
            arr2(&[
                [-0.1425438531921371, 0.0175438531921371, -0.1962719265960686],
                [0.289167038698797, -0.1428867851718394, 0.2908637728763561],
            ])
        );

        assert_rel_eq_arr2!(
            dw,
            arr2(&[
                [0.1074561468078629, -0.125],
                [0.0503346050488132, 0.0837802535269576],
                [-0.0554248075814905, 0.1356401267634788],
                [0.1040626784527447, 0.0212802535269576],
            ])
        );
    }
}
