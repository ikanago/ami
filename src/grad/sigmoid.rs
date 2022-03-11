use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{Dimension, Zip};

use crate::grad::{send_gradient, Function, Tensor};

#[derive(Clone)]
pub struct Sigmoid<D, I>
where
    D: Dimension,
    I: Function,
{
    data: Rc<RefCell<Tensor<D>>>,
    input: I,
    gradient: Rc<RefCell<Tensor<D>>>,
    buffer_for_backward: RefCell<Tensor<D>>,
}

pub fn sigmoid<D, I>(input: &I) -> Sigmoid<D, I>
where
    D: Dimension,
    I: Function<Dim = D>,
{
    let shape = input.data().raw_dim();
    Sigmoid {
        data: Rc::new(RefCell::new(Tensor::zeros(shape.clone()))),
        input: input.clone(),
        gradient: Rc::new(RefCell::new(Tensor::zeros(shape.clone()))),
        buffer_for_backward: RefCell::new(Tensor::zeros(shape)),
    }
}

fn compute_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl<D, I> Function for Sigmoid<D, I>
where
    D: Dimension,
    I: Function<Dim = D>,
{
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

    fn forward(&self) {
        self.input.forward();
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.input.data())
            .for_each(|data, &input| *data = compute_sigmoid(input));
    }

    fn backward(&self) {
        Zip::from(&mut *self.buffer_for_backward.borrow_mut())
            .and(&*self.gradient())
            .and(&*self.input.data())
            .for_each(|buffer, &grad, &input| {
                let s = compute_sigmoid(input);
                *buffer = grad * (s * (1.0 - s));
            });
        send_gradient(&self.input, &*self.buffer_for_backward.borrow());
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
    fn compute_sigmoid() {
        let x = Variable::new(arr2(&[[-2.0, -1.0, 0.0, 1.0, 2.0]])).requires_grad();
        let y = sigmoid(&x);
        y.forward();
        assert_rel_eq_arr2!(
            arr2(&[[
                0.1192029220221175,
                0.2689414213699951,
                0.5000000000000000,
                0.7310585786300049,
                0.8807970779778823,
            ]]),
            y.data().clone()
        );

        y.init_grad();
        y.backward();
        assert_rel_eq_arr2!(arr2(&[[
            0.1049935854035065,
            0.1966119332414819,
            0.2500000000000000,
            0.1966119332414819,
            0.1049935854035066,
        ]]), x.gradient().clone());
    }
}
