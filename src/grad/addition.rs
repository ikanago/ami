use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{DimMax, Dimension, Zip};

use crate::grad::{broadcast_zeros, BroadTensor, Function, Tensor};

use super::{reduce, Broadcasted};

pub fn add<Lhs, Rhs>(lhs: &Lhs, rhs: &Rhs) -> Addition<Lhs, Rhs>
where
    Lhs: Function,
    Rhs: Function,
    Lhs::Dim: DimMax<Rhs::Dim>,
{
    Addition::new(lhs, rhs)
}

#[derive(Clone)]
pub struct Addition<Lhs, Rhs>
where
    Lhs: Function,
    Rhs: Function,
    Lhs::Dim: DimMax<Rhs::Dim>,
{
    data: Rc<RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>>,
    lhs: Lhs,
    rhs: Rhs,
    gradient: Rc<RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>>,
}

impl<Lhs, Rhs> Addition<Lhs, Rhs>
where
    Lhs: Function,
    Rhs: Function,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(lhs: &Lhs, rhs: &Rhs) -> Self {
        let data = broadcast_zeros(&lhs.data(), &rhs.data());
        let shape = data.raw_dim();

        Self {
            data: Rc::new(RefCell::new(data)),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            gradient: Rc::new(RefCell::new(Tensor::zeros(shape))),
        }
    }
}

impl<Lhs, Rhs> Function for Addition<Lhs, Rhs>
where
    Lhs: Function,
    Rhs: Function,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;
    type GradDim = Broadcasted<Lhs::Dim, Rhs::Dim>;

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
            .and_broadcast(&*self.lhs.data())
            .and_broadcast(&*self.rhs.data())
            .for_each(|data, l, r| *data = l + r);
    }

    fn backward(&self) {
        let lhs_grad = reduce(&self.gradient(), self.lhs.gradient().raw_dim());
        self.lhs.update_gradient(&lhs_grad);
        let rhs_grad = reduce(&self.gradient(), self.rhs.gradient().raw_dim());
        self.rhs.update_gradient(&rhs_grad);

        self.lhs.backward();
        self.rhs.backward();
    }
}

#[cfg(test)]
mod tests {
    use crate::{assert_rel_eq_arr2, grad::Variable};

    use super::*;

    use approx::assert_relative_eq;
    use ndarray::{arr1, arr2};

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

    #[test]
    fn broadcast_add() {
        let a = Variable::new(arr2(&[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))
            .requires_grad();
        let b = Variable::new(arr1(&[4.0, 4.0, 4.0])).requires_grad();
        let z = add(&a, &b);
        z.forward();
        assert_rel_eq_arr2!(
            arr2(&[[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]]),
            z.data().clone()
        );

        z.init_gradient();
        z.backward();
        assert_rel_eq_arr2!(Tensor::ones((3, 3)), a.gradient().clone());
        assert_rel_eq_arr2!(arr1(&[3.0, 3.0, 3.0]), b.gradient().clone());
    }
}
