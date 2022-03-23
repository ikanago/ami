use std::marker::PhantomData;

use ndarray::{Array, Dimension, Ix1, Ix2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use crate::grad::{self, add, matmul, relu, Addition, Function, MatrixMultiplication, Variable};

/// Trait to represent learning model.
pub trait Model<D>
where
    D: Dimension,
{
    type In: Function;
    type Out: Function;

    fn forward(&self, input: Variable<D>) -> Self::Out;

    fn update_parameters(&self);
}

pub trait Chainable<D, Prev>
where
    D: Dimension,
    Prev: Model<D>,
{
    type Chained;

    fn chain(self, previous: Prev) -> Self::Chained;
}

pub struct Input<D: Dimension> {
    _dim: PhantomData<D>,
}

impl<D: Dimension> Default for Input<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D> Input<D>
where
    D: Dimension,
{
    pub fn new() -> Self {
        Input { _dim: PhantomData }
    }
}

impl<D> Model<D> for Input<D>
where
    D: Dimension,
{
    type In = Variable<D>;
    type Out = Variable<D>;

    fn forward(&self, input: Variable<D>) -> Self::Out {
        input
    }

    fn update_parameters(&self) {}
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

impl<D, In, Prev> Model<D> for Linear<Prev>
where
    D: Dimension,
    In: Function<Dim = Ix2, GradDim = Ix2>,
    Prev: Model<D, Out = In>,
{
    type In = In;
    type Out = Addition<MatrixMultiplication<In, Variable<Ix2>>, Variable<Ix1>>;

    fn forward(&self, input: Variable<D>) -> Self::Out {
        let previous_output = self.previous.forward(input);
        add(&matmul(&previous_output, &self.weight), &self.bias)
    }

    fn update_parameters(&self) {
        self.previous.update_parameters();
    }
}

impl<D, Prev> Chainable<D, Prev> for Linear<()>
where
    D: Dimension,
    Prev: Model<D>,
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

impl<D, In, Prev> Model<D> for Relu<Prev>
where
    D: Dimension,
    In: Function<Dim = D, GradDim = D>,
    Prev: Model<D, Out = In>,
{
    type In = In;
    type Out = grad::Relu<D, In>;

    fn forward(&self, input: Variable<D>) -> Self::Out {
        let previous_output = self.previous.forward(input);
        relu(&previous_output)
    }

    fn update_parameters(&self) {
        self.previous.update_parameters();
    }
}

impl<D, Prev> Chainable<D, Prev> for Relu<()>
where
    D: Dimension,
    Prev: Model<D>,
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
        let model = sequential![Input::new(), Linear::new(2, 4), Relu::new()];
        let x = Variable::new(arr2(&[[1.0, 0.5], [2.0, 3.0]]));
        model.forward(x);
    }
}
