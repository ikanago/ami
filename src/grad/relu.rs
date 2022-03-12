use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{Dimension, Zip};

use crate::grad::{Function, Tensor};

#[derive(Clone)]
pub struct Relu<D, I>
where
    D: Dimension,
    I: Function,
{
    data: Rc<RefCell<Tensor<D>>>,
    input: I,
    gradient: Rc<RefCell<Tensor<D>>>,
    buffer_for_backward: RefCell<Tensor<D>>,
}

pub fn relu<D, I>(input: &I) -> Relu<D, I>
where
    D: Dimension,
    I: Function<Dim = D>,
{
    let shape = input.data().raw_dim();
    Relu {
        data: Rc::new(RefCell::new(Tensor::zeros(shape.clone()))),
        input: input.clone(),
        gradient: Rc::new(RefCell::new(Tensor::zeros(shape.clone()))),
        buffer_for_backward: RefCell::new(Tensor::zeros(shape)),
    }
}

impl<D, I> Function for Relu<D, I>
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
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.input.data())
            .for_each(|data, &input| *data = if input >= 0.0 { input } else { 0.0 });
    }

    fn backward(&self) {
        Zip::from(&mut *self.buffer_for_backward.borrow_mut())
            .and(&*self.gradient())
            .and(&*self.input.data())
            .for_each(|buffer, &grad, &input| {
                *buffer = grad * if input >= 0.0 { 1.0 } else { 0.0 }
            });
        self.input
            .update_gradient(&*self.buffer_for_backward.borrow());
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
    fn compute_relu() {
        let x = Variable::new(arr2(&[[1.0, 0.0], [2.0, -1.0]])).requires_grad();
        let y = relu(&x);
        y.forward();
        assert_rel_eq_arr2!(arr2(&[[1.0, 0.0], [2.0, 0.0]]), y.data().clone());

        y.init_gradient();
        y.backward();
        assert_rel_eq_arr2!(arr2(&[[1.0, 1.0], [1.0, 0.0]]), x.gradient().clone());
    }
}
