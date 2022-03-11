use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{Dimension, Zip};

use crate::grad::{send_gradient, Function, Tensor};

pub fn mul<D, Lhs, Rhs>(lhs: &Lhs, rhs: &Rhs) -> Multiplication<D, Lhs, Rhs>
where
    D: Dimension,
    Lhs: Function<Dim = D>,
    Rhs: Function<Dim = D>,
{
    Multiplication::new(lhs, rhs)
}

#[derive(Clone)]
pub struct Multiplication<D, Lhs, Rhs>
where
    D: Dimension,
    Lhs: Function,
    Rhs: Function,
{
    // TODO: this cannot restrict the shape of rhs.
    data: Rc<RefCell<Tensor<D>>>,
    lhs: Lhs,
    rhs: Rhs,
    gradient: Rc<RefCell<Tensor<D>>>,
    buffer_for_backward: RefCell<Tensor<D>>,
}

impl<D, Lhs, Rhs> Multiplication<D, Lhs, Rhs>
where
    D: Dimension,
    Lhs: Function<Dim = D>,
    Rhs: Function<Dim = D>,
{
    pub fn new(lhs: &Lhs, rhs: &Rhs) -> Self {
        assert_eq!(lhs.data().shape(), rhs.data().shape());

        Self {
            data: Rc::new(RefCell::new(Tensor::zeros(lhs.data().raw_dim()))),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            gradient: Rc::new(RefCell::new(Tensor::zeros(lhs.data().raw_dim()))),
            buffer_for_backward: RefCell::new(Tensor::zeros(lhs.data().raw_dim())),
        }
    }
}

impl<D, Lhs, Rhs> Function for Multiplication<D, Lhs, Rhs>
where
    D: Dimension,
    Lhs: Function<Dim = D>,
    Rhs: Function<Dim = D>,
{
    type Dim = D;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn gradient(&self) -> Ref<Tensor<D>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn forward(&self) {
        self.lhs.forward();
        self.rhs.forward();

        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.lhs.data())
            .and(&*self.rhs.data())
            .for_each(|data, l, r| *data = l * r);
    }

    fn backward(&self) {
        Zip::from(&mut *self.buffer_for_backward.borrow_mut())
            .and(&*self.gradient())
            .and(&*self.rhs.data())
            .for_each(|buffer, grad, rhs| *buffer = grad * rhs);
        send_gradient(&self.lhs, &*self.buffer_for_backward.borrow());

        Zip::from(&mut *self.buffer_for_backward.borrow_mut())
            .and(&*self.gradient())
            .and(&*self.lhs.data())
            .for_each(|buffer, grad, lhs| *buffer = grad * lhs);
        send_gradient(&self.rhs, &*self.buffer_for_backward.borrow());

        self.lhs.backward();
        self.rhs.backward();
    }
}

#[cfg(test)]
mod tests {
    use crate::{assert_rel_eq_arr2, grad::Variable};

    use super::*;

    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn var_mul_var() {
        let x = Variable::new(arr2(&[[1.0, 0.0], [2.0, -1.0]])).requires_grad();
        let y = Variable::new(arr2(&[[1.0, 1.0], [3.0, -2.0]])).requires_grad();
        let z = mul(&x, &y);
        z.forward();
        assert_rel_eq_arr2!(arr2(&[[1.0, 0.0], [6.0, 2.0]]), z.data().clone());

        z.init_grad();
        z.backward();
        assert_rel_eq_arr2!(arr2(&[[1.0, 0.0], [2.0, -1.0]]), y.gradient().clone());
    }

    #[test]
    fn power_of_2() {
        let x = Variable::new(arr2(&[[1.0, -1.0], [2.0, -3.0]])).requires_grad();
        let z = mul(&x, &x);
        z.forward();
        assert_rel_eq_arr2!(arr2(&[[1.0, 1.0], [4.0, 9.0]]), z.data().clone());

        z.init_grad();
        z.backward();
        assert_rel_eq_arr2!(arr2(&[[2.0, -2.0], [4.0, -6.0]]), x.gradient().clone());
    }

    #[test]
    fn nested_power_of_3() {
        let x = Variable::new(arr2(&[[1.0, -1.0], [2.0, -3.0]])).requires_grad();
        let y = mul(&x, &x);
        let z = mul(&x, &y);
        z.forward();
        assert_rel_eq_arr2!(arr2(&[[1.0, 1.0], [4.0, 9.0]]), y.data().clone());
        assert_rel_eq_arr2!(arr2(&[[1.0, -1.0], [8.0, -27.0]]), z.data().clone());

        z.init_grad();
        z.backward();
        assert_rel_eq_arr2!(arr2(&[[3.0, 3.0], [12.0, 27.0]]), x.gradient().clone());
    }

    #[test]
    fn grad_wrt_const_is_zero() {
        let x = Variable::new(arr2(&[[1.0, 0.0], [2.0, -1.0]])).requires_grad();
        let a = Variable::new(arr2(&[[1.0, 1.0], [3.0, -2.0]]));
        let z = mul(&x, &a);
        z.forward();
        assert_rel_eq_arr2!(arr2(&[[1.0, 0.0], [6.0, 2.0]]), z.data().clone());

        z.init_grad();
        z.backward();
        assert_rel_eq_arr2!(arr2(&[[0.0, 0.0], [0.0, 0.0]]), a.gradient().clone());
        assert_rel_eq_arr2!(arr2(&[[1.0, 1.0], [3.0, -2.0]]), x.gradient().clone());
    }
}
