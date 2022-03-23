use ndarray::{Array, Dimension, Ix1, Ix2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use crate::{
    grad::{self, add, matmul, relu, Addition, Function, MatrixMultiplication, Variable},
    optimizer::Optimizer,
};

/// Trait to represent learning model.
/// The purpose of this trait is to track of learning parameter.
///
/// This trait is a higher rank representation of a computational graph.
/// Constructing a neural network takes 2 steps: `chain` and `forward`.
/// First, consequtive calls of `Chainable::chain` creates a linearly connected `Model`s.
/// Then feed `input` by `Model::forward` for each input batch to obtain a diffrentiable
/// computational graph.
pub trait Model<InD>
where
    InD: Dimension,
{
    type Output: Function;

    /// Construct computational graph while passing `input` to the previous layer of the model.
    /// This process does not carry out actual computation.
    fn forward(&self, input: Variable<InD>) -> Self::Output;

    fn update_parameters<Opt: Optimizer>(&self, optimizer: &Opt);

    fn zero_gradient(&self);
}

pub trait Chainable<InD, Prev>
where
    InD: Dimension,
    Prev: Model<InD>,
{
    type Chained: Model<InD>;

    fn chain(self, previous: Prev) -> Self::Chained;
}

/// Marker struct to specify the end of the propagation of `input` in `Model::forward`.
pub struct Input;

impl Default for Input {
    fn default() -> Self {
        Self
    }
}

impl<InD> Model<InD> for Input
where
    InD: Dimension,
{
    type Output = Variable<InD>;

    fn forward(&self, input: Variable<InD>) -> Self::Output {
        input
    }

    fn update_parameters<Opt: Optimizer>(&self, _optimizer: &Opt) {}

    fn zero_gradient(&self) {}
}

/// Linear transformation layer.
pub struct Linear<Prev> {
    weight: Variable<Ix2>,
    bias: Variable<Ix1>,
    previous: Prev,
}

impl Linear<()> {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = Variable::new(Array::random(
            (in_features, out_features),
            Uniform::new(-1.0, 1.0),
        ))
        .requires_grad();
        let bias =
            Variable::new(Array::random((out_features,), Uniform::new(-1.0, 1.0))).requires_grad();

        Self {
            weight,
            bias,
            previous: (),
        }
    }
}

impl<InD, Input, Prev> Model<InD> for Linear<Prev>
where
    InD: Dimension,
    Input: Function<Dim = Ix2, GradDim = Ix2>,
    Prev: Model<InD, Output = Input>,
{
    type Output = Addition<MatrixMultiplication<Input, Variable<Ix2>>, Variable<Ix1>>;

    fn forward(&self, input: Variable<InD>) -> Self::Output {
        let previous_output = self.previous.forward(input);
        add(&matmul(&previous_output, &self.weight), &self.bias)
    }

    fn update_parameters<Opt: Optimizer>(&self, optimizer: &Opt) {
        optimizer.update(&self.weight);
        optimizer.update(&self.bias);
        self.previous.update_parameters(optimizer);
    }

    fn zero_gradient(&self) {
        self.weight.zero_gradient();
        self.bias.zero_gradient();
        self.previous.zero_gradient();
    }
}

impl<InD, Prev> Chainable<InD, Prev> for Linear<()>
where
    InD: Dimension,
    Prev: Model<InD>,
    Prev::Output: Function<Dim = Ix2, GradDim = Ix2>,
{
    type Chained = Linear<Prev>;

    fn chain(self, previous: Prev) -> Self::Chained {
        Self::Chained {
            weight: self.weight,
            bias: self.bias,
            previous,
        }
    }
}

/// Layer applying ReLU.
pub struct Relu<Prev> {
    previous: Prev,
}

impl Relu<()> {
    pub fn new() -> Self {
        Self { previous: () }
    }
}

impl Default for Relu<()> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D, InD, Input, Prev> Model<InD> for Relu<Prev>
where
    D: Dimension,
    InD: Dimension,
    Input: Function<Dim = D, GradDim = D>,
    Prev: Model<InD, Output = Input>,
{
    type Output = grad::Relu<D, Input>;

    fn forward(&self, input: Variable<InD>) -> Self::Output {
        let previous_output = self.previous.forward(input);
        relu(&previous_output)
    }

    fn update_parameters<Opt: Optimizer>(&self, optimizer: &Opt) {
        self.previous.update_parameters(optimizer);
    }

    fn zero_gradient(&self) {
        self.previous.zero_gradient();
    }
}

impl<D, InD, Prev> Chainable<InD, Prev> for Relu<()>
where
    D: Dimension,
    InD: Dimension,
    Prev: Model<InD>,
    Prev::Output: Function<Dim = D, GradDim = D>,
{
    type Chained = Relu<Prev>;

    fn chain(self, previous: Prev) -> Self::Chained {
        Self::Chained { previous }
    }
}

#[macro_export]
macro_rules! sequential {
    [$first:expr, $second:expr] => {
        $second.chain($first)
    };

    [$first:expr, $second:expr, $($rest:expr),+] => {
        sequential![$second.chain($first), $($rest),+]
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::arr2;

    #[test]
    fn forward_sequential() {
        let model = sequential![Input, Linear::new(2, 4), Relu::new()];
        let x = Variable::new(arr2(&[[1.0, 0.5], [2.0, 3.0]]));
        model.forward(x);
    }
}
