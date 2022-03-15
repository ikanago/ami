use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::Dimension;

use crate::grad::{Function, Tensor};

#[derive(Clone)]
pub struct Identity<D, I>
where
    D: Dimension,
    I: Function,
{
    data: Rc<RefCell<Tensor<D>>>,
    input: I,
    gradient: Rc<RefCell<Tensor<D>>>,
}

pub fn identity<D, I>(input: &I) -> Identity<D, I>
where
    D: Dimension,
    I: Function<Dim = D>,
{
    let shape = input.data().raw_dim();
    Identity {
        data: Rc::new(RefCell::new(Tensor::zeros(shape.clone()))),
        input: input.clone(),
        gradient: Rc::new(RefCell::new(Tensor::zeros(shape))),
    }
}

impl<D, I> Function for Identity<D, I>
where
    D: Dimension,
    I: Function<Dim = D, GradDim = D>,
{
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

    fn forward(&self) {
        self.input.forward();
        *self.data.borrow_mut() = self.input.data().clone();
    }

    fn backward(&self) {
        self.input.update_gradient(&*self.gradient());
        self.input.backward();
    }
}

#[cfg(test)]
mod tests {
    use crate::{assert_rel_eq_arr2, grad::Variable};

    use super::*;

    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn compute_identity() {
        let x = Variable::new(arr2(&[[1.0, 0.0], [2.0, -1.0]])).requires_grad();
        let y = identity(&x);
        y.forward();
        assert_rel_eq_arr2!(arr2(&[[1.0, 0.0], [2.0, -1.0]]), y.data().clone());

        y.init_gradient();
        y.backward();
        assert_rel_eq_arr2!(arr2(&[[1.0, 1.0], [1.0, 1.0]]), x.gradient().clone());
    }
}
