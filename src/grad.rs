pub mod addition;
pub mod matmul;
pub mod mul;
pub mod relu;
pub mod sigmoid;

use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{Array, ArrayView, Dimension, IntoNdProducer, Zip};

pub type Tensor<D> = Array<f32, D>;

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
    fn forward(&self);

    /// Run backward propagation.
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
