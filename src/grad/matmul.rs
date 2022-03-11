use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{linalg::general_mat_mul, Dimension, Ix2};

use crate::grad::{send_gradient, Function, Tensor};

pub fn matmul<Lhs, Rhs>(lhs: &Rc<Lhs>, rhs: &Rc<Rhs>) -> Rc<MatrixMultiplication<Lhs, Rhs>>
where
    Lhs: Function<Dim = Ix2>,
    Rhs: Function<Dim = Ix2>,
{
    Rc::new(MatrixMultiplication::new(lhs, rhs))
}

pub struct MatrixMultiplication<Lhs, Rhs>
where
    Lhs: Function,
    Rhs: Function,
{
    data: RefCell<Tensor<Ix2>>,
    lhs: Rc<Lhs>,
    rhs: Rc<Rhs>,
    gradient: RefCell<Tensor<Ix2>>,
}

impl<Lhs, Rhs> MatrixMultiplication<Lhs, Rhs>
where
    Lhs: Function<Dim = Ix2>,
    Rhs: Function<Dim = Ix2>,
{
    pub fn new(lhs: &Rc<Lhs>, rhs: &Rc<Rhs>) -> Self {
        Self {
            data: RefCell::new(Tensor::zeros((lhs.data().nrows(), rhs.data().ncols()))),
            lhs: Rc::clone(lhs),
            rhs: Rc::clone(rhs),
            gradient: RefCell::new(Tensor::zeros(lhs.data().raw_dim())),
        }
    }

    pub fn init_grad(&self) {
        *self.gradient.borrow_mut() = Tensor::ones(self.lhs.data().raw_dim());
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

    fn backward(&self) {}
}

#[cfg(test)]
mod tests {
    use crate::{assert_rel_eq_arr2, grad::Variable};

    use super::*;

    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn forward_mat_mul() {
        let w = arr2(&[[1.0, 2.0], [-1.0, 3.0], [1.0, -1.0]]);
        let x = arr2(&[[-1.0, 3.0, 2.0], [4.0, 1.0, 0.5]]);
        let expected = x.dot(&w);

        let w = Rc::new(Variable::new(w));
        let x = Rc::new(Variable::new(x).requires_grad());
        let z = matmul(&x, &w);
        z.forward();
        assert_rel_eq_arr2!(z.data().clone(), expected);
    }
}
