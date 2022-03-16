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

use ndarray::{Array, ArrayD, ArrayView, Axis, DimMax, Dimension, IntoNdProducer, Zip};

pub type Tensor<D> = Array<f32, D>;
pub type DynTensor = ArrayD<f32>;
pub(crate) type Broadcasted<LhsDim, RhsDim> = <LhsDim as DimMax<RhsDim>>::Output;
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

/// Create zero tensor whose shape is broadcasted one from lhs and rhs.
///
/// Broadcast algorithm:
/// * Pad the smaller shape with 1 to match the dimentionan to the larger shape.
/// * Iterate each dimention size.
///   * Both dimention size match, adopt it.
///   * One of the dimention size is 1, adopt the other.
///   * Both dimention size is different and not 1, these shape cannot be broadcasted.
pub fn broadcast_zeros<LhsDim, RhsDim>(
    lhs: &Tensor<LhsDim>,
    rhs: &Tensor<RhsDim>,
) -> BroadTensor<LhsDim, RhsDim>
where
    LhsDim: Dimension + DimMax<RhsDim>,
    RhsDim: Dimension,
{
    let (larger_shape, smaller_shape) = if lhs.ndim() > rhs.ndim() {
        (lhs.shape(), rhs.shape())
    } else {
        (rhs.shape(), lhs.shape())
    };
    let leading_dims = larger_shape.len() - smaller_shape.len();

    let mut broadcasted_shape = Broadcasted::<LhsDim, RhsDim>::zeros(larger_shape.len());
    let broadcasted_shape_iter = broadcasted_shape.slice_mut().iter_mut();
    let smaller_shape_iter = [1]
        .iter()
        .cycle()
        .take(leading_dims)
        .chain(smaller_shape)
        .collect::<Vec<_>>();
    larger_shape
        .iter()
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

/// Sum up along specified `axis` in-place.
/// This reduces the dimentionality of `x` by 1.
fn sum_axis_inplace(x: &mut DynTensor, axis: Axis) {
    let (accumulated, rest) = x.view_mut().split_at(axis, 1);
    Zip::from(accumulated.remove_axis(axis))
        .and(rest.lanes(axis))
        .for_each(|acc, rest| *acc += rest.sum());
    // There remains elements refered by `rest`, so remove them.
    x.index_axis_inplace(axis, 0);
}

/// Reduce the dimentionality of the tensor `x` to `dim`.
/// In the process, sum up along the axis which lacks or whose size is one.
pub fn reduce<D, E>(x: &Tensor<D>, dim: E) -> Tensor<E>
where
    D: Dimension,
    E: Dimension,
{
    // Convert to dynamic array because following process reduces the dimentionality in a loop.
    let mut x = x.clone().into_dyn();

    while x.ndim() > dim.ndim() {
        sum_axis_inplace(&mut x, Axis(0));
    }

    for (axis, _) in dim
        .slice()
        .iter()
        .enumerate()
        .filter(|(_, &size)| size == 1)
    {
        sum_axis_inplace(&mut x, Axis(axis));
        x.insert_axis_inplace(Axis(axis));
    }

    assert_eq!(x.raw_dim(), dim.into_dyn());
    x.into_dimensionality::<E>().unwrap()
}

#[cfg(test)]
mod tests {
    use crate::assert_rel_eq_arr2;

    use super::*;

    use approx::assert_relative_eq;
    use ndarray::{arr1, arr2, IntoDimension};

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

    #[test]
    fn reduce_2d_to_2d() {
        let a = arr2(&[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]);
        let a_3_1 = reduce(&a, IntoDimension::into_dimension((3, 1)));
        assert_rel_eq_arr2!(arr2(&[[3.0], [6.0], [9.0]]), a_3_1);

        let a_1_3 = reduce(&a, IntoDimension::into_dimension((1, 3)));
        assert_rel_eq_arr2!(arr2(&[[6.0, 6.0, 6.0]]), a_1_3);
    }

    #[test]
    fn reduce_2d_to_1d() {
        let a = arr2(&[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]);
        let a_3 = reduce(&a, IntoDimension::into_dimension((3,)));
        assert_rel_eq_arr2!(arr1(&[6.0, 6.0, 6.0]), a_3);
    }
}
