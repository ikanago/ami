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
        let weights = Array::random((output_dim, input_dim), Uniform::new(0.0, 1.0));
        Linear::with_weights(input_dim, output_dim, activation, weights)
    }

    pub fn with_weights(
        input_dim: usize,
        output_dim: usize,
        activation: Sigmoid,
        weights: Array2<f64>,
    ) -> Self {
        let inputs = Array2::zeros((input_dim, 1));
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
        let weights_derivative = dot_products_derivative.dot(&self.inputs.t());
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
        let weights = arr2(&[[1.0, -1.0, 0.5], [2.0, -1.0, 2.0]]);
        let mut layer = Linear::with_weights(3, 2, Sigmoid, weights);
        let inputs = arr2(&[[1.0], [0.5], [-0.5]]);
        let outputs = layer.forward(inputs);
        assert_rel_eq_arr2!(outputs, arr2(&[[0.5621765008857981], [0.6224593312018546]]));

        let train = arr2(&[[1.0], [0.0]]);
        let error = train - outputs;
        let (dz, dw) = layer.backward(&error.view());

        assert_rel_eq_arr2!(
            dz,
            arr2(&[
                [-0.1847972216984755],
                [0.0385169681715178],
                [-0.2386788643761953]
            ])
        );

        assert_rel_eq_arr2!(
            dw,
            arr2(&[
                [0.1077632853554398, 0.0538816426777199, -0.0538816426777199,],
                [-0.1462802535269576, -0.0731401267634788, 0.0731401267634788,]
            ])
        );
    }
}
