use ndarray::{Array, Array2, ArrayView2, Axis};
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
        let weights = Array::random((output_dim, input_dim), Uniform::new(0.0, 1.0));
        Linear::with_weights(input_dim, output_dim, activation, weights)
    }

    pub fn with_weights(
        input_dim: usize,
        output_dim: usize,
        activation: Sigmoid,
        weights: Array2<f64>,
    ) -> Self {
        let inputs = Array2::zeros((input_dim + 1, 1));
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
        let ones = Array::<f64, _>::ones((1, 1));
        let mut inputs = inputs;
        inputs.append(Axis(0), ones.view()).unwrap();

        self.inputs = inputs;
        self.dot_products = self.weights.dot(&self.inputs);
        self.outputs = self.activation.compute(&self.dot_products);
        self.outputs.view()
    }

    pub fn backward(&mut self, errors: &ArrayView2<f64>) -> (Array2<f64>, Array2<f64>) {
        let activation_derivative = self.activation.derivative(&self.dot_products);
        let dot_products_derivative = activation_derivative * errors;
        let mut inputs_derivative = self.weights.t().dot(&dot_products_derivative);
        let weights_derivative = dot_products_derivative.dot(&self.inputs.t());

        inputs_derivative.remove_index(Axis(0), inputs_derivative.nrows() - 1);
        (inputs_derivative, weights_derivative)
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_rel_eq_arr2;

    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn layer_forward() {
        let weights = arr2(&[[1.0, -1.0, 0.5, -2.0], [2.0, -1.0, 2.0, -0.5]]);
        let mut layer = Linear::with_weights(3, 2, Sigmoid, weights);
        let inputs = arr2(&[[1.0], [0.5], [-0.5]]);
        let outputs = layer.forward(inputs);
        assert_rel_eq_arr2!(outputs, arr2(&[[0.1480471980316895], [0.5]]));

        let train = arr2(&[[1.0], [0.0]]);
        let error = train - outputs;
        let (dz, dw) = layer.backward(&error.view());

        assert_rel_eq_arr2!(
            dz,
            arr2(&[
                [-0.1425438531921371],
                [0.0175438531921371],
                [-0.1962719265960686],
            ])
        );

        assert_rel_eq_arr2!(
            dw,
            arr2(&[
                [
                    0.1074561468078629,
                    0.0537280734039314,
                    -0.0537280734039314,
                    0.1074561468078629
                ],
                [-0.125, -0.0625, 0.0625, -0.125],
            ])
        );
    }
}
