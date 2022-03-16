pub mod addition;
pub mod identity;
pub mod matmul;
pub mod mse;
pub mod mul;
pub mod relu;
pub mod sigmoid;

use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{Array, ArrayView, DimMax, Dimension, IntoNdProducer, Zip};

pub type Tensor<D> = Array<f32, D>;
pub(crate) type Broadcasted<Lhs, Rhs> = <Lhs as DimMax<Rhs>>::Output;
pub(crate) type BroadTensor<Lhs, Rhs> = Tensor<Broadcasted<Lhs, Rhs>>;

/// Trait to represent a computational graph of a function to be diffrentiated.
/// All node in the graph implements this trait.
pub trait Function: Clone {
    type Dim: Dimension;
    type GradDim: Dimension;

    /// Return the reference to this node's value which has computed in forward path for given inputs.
    fn data(&self) -> Ref<Tensor<Self::Dim>>;

    /// Return the reference to the gradient of the whole function with respect to this node.
    fn gradient(&self) -> Ref<Tensor<Self::GradDim>>;

    /// Return the mutable reference to the gradient of the whole function with respect to this node.
    fn gradient_mut(&self) -> RefMut<Tensor<Self::GradDim>>;

    /// Initialize gradient with the tensor whose elements are all 1.0.
    /// This is called when the struct instance is the root of the computation graph.
    fn init_gradient(&self) {
        let shape = self.gradient().raw_dim();
        *self.gradient_mut() = Tensor::ones(shape);
    }

    /// Update the gradient by adding `gradient`.
    fn update_gradient<'a, P>(&self, gradient: P)
    where
        P: IntoNdProducer<
            Dim = Self::GradDim,
            Output = ArrayView<'a, f32, Self::GradDim>,
            Item = &'a f32,
        >,
    {
        Zip::from(&mut *self.gradient_mut())
            .and(gradient.into_producer())
            .for_each(|grad, &incoming| *grad += incoming);
    }

    /// Run forward propagation.
    /// Call `forward` function(s) of the children node(s) before computing this node's data.
    fn forward(&self);

    /// Run backward propagation.
    /// Call `forward` function(s) of the children node(s) after computing this node's gradient.
    fn backward(&self);
}

/// Struct to represent a variable in a function.
/// By default, the function cannot be diffrentiated with respect to this variable.
/// To diffrentiate, call `requires_grad()`.
#[derive(Clone)]
pub struct Variable<D> {
    data: Rc<RefCell<Tensor<D>>>,
    gradient: Rc<RefCell<Tensor<D>>>,
    requires_grad: bool,
}

impl<D> Variable<D>
where
    D: Dimension,
{
    pub fn new(data: Tensor<D>) -> Self {
        let shape = data.raw_dim();

        Self {
            data: Rc::new(RefCell::new(data)),
            gradient: Rc::new(RefCell::new(Tensor::zeros(shape))),
            requires_grad: false,
        }
    }

    pub fn requires_grad(self) -> Self {
        Self {
            requires_grad: true,
            ..self
        }
    }

    pub fn data_mut(&self) -> RefMut<Tensor<D>> {
        self.data.borrow_mut()
    }
}

impl<D: Dimension> Function for Variable<D> {
    type Dim = D;
    type GradDim = D;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn gradient(&self) -> Ref<Tensor<Self::GradDim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::GradDim>> {
        self.gradient.borrow_mut()
    }

    fn update_gradient<'a, P>(&self, gradient: P)
    where
        P: IntoNdProducer<
            Dim = Self::GradDim,
            Output = ArrayView<'a, f32, Self::GradDim>,
            Item = &'a f32,
        >,
    {
        if !self.requires_grad {
            return;
        }

        Zip::from(&mut *self.gradient_mut())
            .and(gradient.into_producer())
            .for_each(|grad, &incoming| *grad += incoming);
    }

    fn forward(&self) {}

    fn backward(&self) {}
}

pub(crate) fn broadcast_zeros<Lhs, Rhs>(
    lhs: &Tensor<Lhs>,
    rhs: &Tensor<Rhs>,
) -> BroadTensor<Lhs, Rhs>
where
    Lhs: Dimension + DimMax<Rhs>,
    Rhs: Dimension,
{
    let (larger_shape, smaller_shape) = if lhs.ndim() > rhs.ndim() {
        (lhs.shape(), rhs.shape())
    } else {
        (rhs.shape(), lhs.shape())
    };
    let leading_dims = larger_shape.len() - smaller_shape.len();

    let mut broadcasted_shape = Broadcasted::<Lhs, Rhs>::zeros(larger_shape.len());
    let broadcasted_shape_iter = broadcasted_shape.slice_mut().iter_mut().rev();
    let smaller_shape_iter = smaller_shape
        .iter()
        .rev()
        .chain([1].iter().cycle().take(leading_dims))
        .collect::<Vec<_>>();
    larger_shape
        .iter()
        .rev()
        .zip(smaller_shape_iter)
        .zip(broadcasted_shape_iter)
        .for_each(|((&l, &r), res)| {
            *res = if l == r {
                l
            } else if l == 1 {
                r
            } else if r == 1 {
                l
            } else {
                panic!("Shape does not met: {:?}, {:?}", lhs.shape(), rhs.shape());
            }
        });

    Tensor::zeros(broadcasted_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_shape() {
        let a = Tensor::zeros((2, 3, 4));
        let b = Tensor::zeros((2, 3, 4));
        let x = broadcast_zeros(&a, &b);
        assert_eq!(&[2, 3, 4], x.shape());
    }

    #[test]
    fn one_of_dim_has_one() {
        let a = Tensor::zeros((2, 3, 4));
        let b = Tensor::zeros((2, 1, 4));
        let x = broadcast_zeros(&a, &b);
        assert_eq!(&[2, 3, 4], x.shape());
    }

    #[test]
    fn some_of_dim_has_one_for_a_shape() {
        let a = Tensor::zeros((2, 3, 4));
        let b = Tensor::zeros((2, 1, 1));
        let x = broadcast_zeros(&a, &b);
        assert_eq!(&[2, 3, 4], x.shape());
    }

    #[test]
    fn some_of_dim_has_one_for_either_shape() {
        let a = Tensor::zeros((2, 1, 4, 1));
        let b = Tensor::zeros((2, 1, 1, 5));
        let x = broadcast_zeros(&a, &b);
        assert_eq!(&[2, 1, 4, 5], x.shape());
    }

    #[test]
    fn one_of_shape_lacks_one_dim() {
        let a = Tensor::zeros((2, 3, 4));
        let b = Tensor::zeros((1, 4));
        let x = broadcast_zeros(&a, &b);
        assert_eq!(&[2, 3, 4], x.shape());
    }

    #[test]
    #[should_panic]
    fn one_of_dim_is_different_non_one() {
        let a = Tensor::zeros((2, 3, 4));
        let b = Tensor::zeros((2, 2, 4));
        broadcast_zeros(&a, &b);
    }

    #[test]
    #[should_panic]
    fn one_of_dim_is_different_non_one_and_one_of_shape_lacks_one_dim() {
        let a = Tensor::zeros((2, 3, 4));
        let b = Tensor::zeros((2, 4));
        broadcast_zeros(&a, &b);
    }
}
