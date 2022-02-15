pub mod mul;

use std::cell::{Ref, RefCell, RefMut};

use ndarray::{Array, ArrayView, Dimension, IntoNdProducer, Zip};

pub type Tensor<D> = Array<f32, D>;

/// Trait to represent a computational graph of a function to be diffrentiated.
/// All node in the graph implements this trait.
pub trait Function {
    type Dim: Dimension;

    /// Return the reference to this node's value which has computed in forward path for given inputs.
    fn data(&self) -> Ref<Tensor<Self::Dim>>;

    /// Return the reference to the gradient of the whole function with respect to this node.
    fn gradient(&self) -> Ref<Tensor<Self::Dim>>;

    /// Return the mutable reference to the gradient of the whole function with respect to this node.
    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>>;

    /// Run forward propagation.
    fn forward(&self);

    /// Run backward propagation.
    fn backward(&self);
}

/// Struct to represent a variable in a function.
/// By default, the function cannot be diffrentiated with respect to this variable.
/// To diffrentiate, call `requires_grad()`.
#[derive(Clone)]
pub struct Variable<D> {
    data: RefCell<Tensor<D>>,
    gradient: RefCell<Tensor<D>>,
    requires_grad: bool,
}

impl<D> Variable<D>
where
    D: Dimension,
{
    pub fn new(data: Tensor<D>) -> Self {
        let dim = data.raw_dim();

        Self {
            data: RefCell::new(data),
            gradient: RefCell::new(Tensor::zeros(dim)),
            requires_grad: false,
        }
    }

    pub fn requires_grad(self) -> Self {
        Self {
            requires_grad: true,
            ..self
        }
    }
}

impl<D: Dimension> Function for Variable<D> {
    type Dim = D;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn forward(&self) {}

    fn backward(&self) {}
}

pub(crate) fn send_gradient<'a, F, P>(node: &F, gradient: P)
where
    F: Function,
    P: IntoNdProducer<Dim = F::Dim, Output = ArrayView<'a, f32, F::Dim>, Item = &'a f32>,
{
    Zip::from(&mut *node.gradient_mut())
        .and(gradient.into_producer())
        .for_each(|node_grad, grad| *node_grad += *grad);
}
