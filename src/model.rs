use std::marker::PhantomData;

use ndarray::{Array, Ix1, Ix2, Dimension};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use crate::grad::{self, add, matmul, Addition, Function, MatrixMultiplication, Variable, relu, sigmoid};

/// Trait to represent learning model.
pub trait Model<In, Out>
where
    In: Function,
    Out: Function,
{
    fn forward(&self, input: In) -> Out;
}

/// Linear transformation layer.
pub struct Linear {
    weight: Variable<Ix2>,
    bias: Variable<Ix1>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = Variable::new(Array::random(
            (in_features, out_features),
            Uniform::new(-1.0, 1.0),
        ))
        .requires_grad();
        let bias =
            Variable::new(Array::random((out_features,), Uniform::new(-1.0, 1.0))).requires_grad();

        Self { weight, bias }
    }
}

impl<In> Model<In, Addition<MatrixMultiplication<In, Variable<Ix2>>, Variable<Ix1>>> for Linear
where
    In: Function<Dim = Ix2, GradDim = Ix2>,
{
    fn forward(
        &self,
        input: In,
    ) -> Addition<MatrixMultiplication<In, Variable<Ix2>>, Variable<Ix1>> {
        add(&matmul(&input, &self.weight), &self.bias)
    }
}

/// Layer applying ReLU.
pub struct Relu;

impl<D, In> Model<In, grad::Relu<D, In>> for Relu
where
    D: Dimension,
    In: Function<Dim = D, GradDim = D>,
{
    fn forward(&self, input: In) -> grad::Relu<D, In> {
        relu(&input)
    }
}

/// Layer applying sigmoid.
pub struct Sigmoid;

impl<D, In> Model<In, grad::Sigmoid<D, In>> for Sigmoid
where
    D: Dimension,
    In: Function<Dim = D, GradDim = D>,
{
    fn forward(&self, input: In) -> grad::Sigmoid<D, In> {
        sigmoid(&input)
    }
}

pub struct Compose<F, S, In, Mid, Out>
where
    F: Model<In, Mid>,
    S: Model<Mid, Out>,
    In: Function,
    Mid: Function,
    Out: Function,
{
    first: F,
    second: S,
    _in: PhantomData<In>,
    _mid: PhantomData<Mid>,
    _out: PhantomData<Out>,
}

/// Compose two models.
/// Incoming data is applied to `first`, then applied to `second`.
pub fn compose<F, S, In, Mid, Out>(first: F, second: S) -> Compose<F, S, In, Mid, Out>
where
    F: Model<In, Mid>,
    S: Model<Mid, Out>,
    In: Function,
    Mid: Function,
    Out: Function,
{
    Compose {
        first,
        second,
        _in: PhantomData,
        _mid: PhantomData,
        _out: PhantomData,
    }
}

impl<F, S, In, Mid, Out> Model<In, Out> for Compose<F, S, In, Mid, Out>
where
    F: Model<In, Mid>,
    S: Model<Mid, Out>,
    In: Function,
    Mid: Function,
    Out: Function,
{
    fn forward(&self, input: In) -> Out {
        let mid = self.first.forward(input);
        self.second.forward(mid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::arr2;

    #[test]
    fn forward_compose() {
        let model = compose(Linear::new(2, 4), Linear::new(4, 1));
        let x = Variable::new(arr2(&[[1.0, 0.5], [2.0, 3.0]]));
        model.forward(x);
    }
}
