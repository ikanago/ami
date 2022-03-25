mod gradient_descent;

use ndarray::Dimension;

use crate::grad::Variable;

pub use gradient_descent::GradientDescent;

/// Trait to abstruct optimizers.
pub trait Optimizer {
    fn update<D>(&self, parameter: &Variable<D>)
    where
        D: Dimension;
}
