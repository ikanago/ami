use ndarray::Array2;

use crate::layer::Linear;

pub struct Network {
    layers: Vec<Linear>,
    learning_rate: f64,
}

impl Network {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            layers: Vec::new(),
            learning_rate,
        }
    }

    pub fn add_layer(&mut self, layer: Linear) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, inputs: Array2<f64>) -> Array2<f64> {
        let mut outputs = inputs;
        for layer in self.layers.iter_mut() {
            outputs = layer.forward(outputs);
        }
        outputs
    }

    pub fn backward(&mut self, error: Array2<f64>) {
        let mut inputs_derivative = error;
        for layer in self.layers.iter_mut().rev() {
            let (dx, dw) = layer.backward(inputs_derivative);
            inputs_derivative = dx;
            layer.update_weights(self.learning_rate, dw);
        }
    }
}
