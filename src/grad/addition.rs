use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{Dimension, Zip};

use crate::grad::{Function, Tensor};

pub fn add<D, Lhs, Rhs>(lhs: &Lhs, rhs: &Rhs) -> Addition<D, Lhs, Rhs>
where
    D: Dimension,
    Lhs: Function<Dim = D>,
    Rhs: Function<Dim = D>,
{
    Addition::new(lhs, rhs)
}

#[derive(Clone)]
pub struct Addition<D, Lhs, Rhs>
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
}

impl<D, Lhs, Rhs> Addition<D, Lhs, Rhs>
where
    D: Dimension,
    Lhs: Function<Dim = D>,
    Rhs: Function<Dim = D>,
{
    pub fn new(lhs: &Lhs, rhs: &Rhs) -> Self {
        assert_eq!(lhs.data().shape(), rhs.data().shape());

        let shape = lhs.data().raw_dim();

        Self {
            data: Rc::new(RefCell::new(Tensor::zeros(shape.clone()))),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            gradient: Rc::new(RefCell::new(Tensor::zeros(shape))),
        }
    }
}

impl<D, Lhs, Rhs> Function for Addition<D, Lhs, Rhs>
where
    D: Dimension,
    Lhs: Function<Dim = D, GradDim = D>,
    Rhs: Function<Dim = D, GradDim = D>,
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
        self.lhs.forward();
        self.rhs.forward();

        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.lhs.data())
            .and(&*self.rhs.data())
            .for_each(|data, l, r| *data = l + r);
    }

    fn backward(&self) {
        self.lhs.update_gradient(&*self.gradient());
        self.rhs.update_gradient(&*self.gradient());

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
        let z = add(&x, &y);
        z.forward();
        assert_rel_eq_arr2!(arr2(&[[2.0, 1.0], [5.0, -3.0]]), z.data().clone());

        z.init_gradient();
        z.backward();
        assert_rel_eq_arr2!(Tensor::ones((2, 2)), y.gradient().clone());
    }

    #[test]
    fn nested_add() {
        let x = Variable::new(arr2(&[[1.0, -1.0], [2.0, -3.0]])).requires_grad();
        let y = add(&x, &x);
        let z = add(&x, &y);
        z.forward();
        assert_rel_eq_arr2!(arr2(&[[2.0, -2.0], [4.0, -6.0]]), y.data().clone());
        assert_rel_eq_arr2!(arr2(&[[3.0, -3.0], [6.0, -9.0]]), z.data().clone());

        z.init_gradient();
        z.backward();
        assert_rel_eq_arr2!(3.0 * Tensor::ones((2, 2)), x.gradient().clone());
    }
}
