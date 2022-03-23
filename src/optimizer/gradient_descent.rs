use crate::{
    grad::Function,
    grad::{Tensor, Variable},
    optimizer::Optimizer,
};

use ndarray::{Dimension, Zip};

pub struct GradientDescent {
    learning_rate: f32,
}

impl GradientDescent {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for GradientDescent {
    fn update<D>(&self, parameter: &Variable<D>)
    where
        D: Dimension,
    {
        let mut buffer = Tensor::zeros(parameter.data().raw_dim());
        Zip::from(&mut buffer)
            .and(&*parameter.data())
            .and(&*parameter.gradient())
            .for_each(|buffer, d, g| *buffer = d - self.learning_rate * g);
        *parameter.data_mut() = buffer;
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_rel_eq_arr2;

    use super::*;

    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn update_gradient_descent() {
        let w = Variable::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]));
        *w.gradient_mut() = arr2(&[[1.0, -0.5], [0.2, -2.0]]);

        let opt = GradientDescent::new(0.5);
        opt.update(&w);
        assert_rel_eq_arr2!(arr2(&[[0.5, 2.25], [2.9, 5.0]]), w.data().clone());
    }
}
