use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{linalg::general_mat_mul, Ix2};

use crate::grad::{send_gradient, Function, Tensor};

pub fn matmul<Lhs, Rhs>(lhs: &Lhs, rhs: &Rhs) -> MatrixMultiplication<Lhs, Rhs>
where
    Lhs: Function<Dim = Ix2>,
    Rhs: Function<Dim = Ix2>,
{
    MatrixMultiplication::new(lhs, rhs)
}

#[derive(Clone)]
pub struct MatrixMultiplication<Lhs, Rhs>
where
    Lhs: Function,
    Rhs: Function,
{
    data: Rc<RefCell<Tensor<Ix2>>>,
    lhs: Lhs,
    rhs: Rhs,
    gradient: Rc<RefCell<Tensor<Ix2>>>,
}

impl<Lhs, Rhs> MatrixMultiplication<Lhs, Rhs>
where
    Lhs: Function<Dim = Ix2>,
    Rhs: Function<Dim = Ix2>,
{
    pub fn new(lhs: &Lhs, rhs: &Rhs) -> Self {
        let shape = (lhs.data().nrows(), rhs.data().ncols());
        Self {
            data: Rc::new(RefCell::new(Tensor::zeros(shape))),
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            gradient: Rc::new(RefCell::new(Tensor::zeros(shape))),
        }
    }
}

impl<Lhs, Rhs> Function for MatrixMultiplication<Lhs, Rhs>
where
    Lhs: Function<Dim = Ix2>,
    Rhs: Function<Dim = Ix2>,
{
    type Dim = Ix2;

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
        self.lhs.forward();
        self.rhs.forward();
        general_mat_mul(
            1.0,
            &*self.lhs.data(),
            &*self.rhs.data(),
            1.0,
            &mut *self.data.borrow_mut(),
        );
    }

    fn backward(&self) {
        let lhs_grad = self.gradient().dot(&self.rhs.data().t());
        send_gradient(&self.lhs, &lhs_grad);

        // Use `general_mat_mul` because we have to move transposed matrix when computing dot
        // product.
        let mut rhs_grad = Tensor::zeros(self.rhs.data().raw_dim());
        general_mat_mul(
            1.0,
            &self.lhs.data().t(),
            &self.gradient(),
            1.0,
            &mut rhs_grad,
        );
        send_gradient(&self.rhs, &rhs_grad);

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
    fn forward_matmul() {
        let x = arr2(&[[1.0, 2.0], [-1.0, 3.0], [1.0, -1.0]]);
        let w = arr2(&[[-1.0, 3.0, 2.0], [4.0, 1.0, 0.5]]);
        let expected = x.dot(&w);

        let x = Variable::new(x).requires_grad();
        let w = Variable::new(w).requires_grad();
        let z = matmul(&x, &w);
        z.forward();
        assert_rel_eq_arr2!(z.data().clone(), expected);
    }

    #[test]
    fn backward_matmul() {
        let x = Variable::new(arr2(&[[1.0, 2.0], [-1.0, 3.0], [1.0, -1.0]])).requires_grad();
        let w = Variable::new(arr2(&[[-1.0, 3.0, 2.0], [4.0, 1.0, 0.5]])).requires_grad();
        let z = matmul(&x, &w);
        z.forward();

        z.init_grad();
        z.backward();
        assert_rel_eq_arr2!(
            arr2(&[[4.0, 5.5], [4.0, 5.5], [4.0, 5.5]]),
            x.gradient().clone()
        );
        assert_rel_eq_arr2!(
            arr2(&[[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]]),
            w.gradient().clone()
        );
    }
}
