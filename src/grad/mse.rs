use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{arr0, Dimension, Ix0, Zip};

use crate::grad::{Function, Tensor};

pub fn mse<D, Input, Target>(input: &Input, target: &Target) -> MeanSquaredError<D, Input, Target>
where
    D: Dimension,
    Input: Function<Dim = D>,
    Target: Function<Dim = D>,
{
    MeanSquaredError::new(input, target)
}

#[derive(Clone)]
pub struct MeanSquaredError<D, Input, Target>
where
    D: Dimension,
    Input: Function,
    Target: Function,
{
    // TODO: this cannot restrict the shape of target.
    data: Rc<RefCell<Tensor<Ix0>>>,
    input: Input,
    target: Target,
    gradient: Rc<RefCell<Tensor<D>>>,
    buffer_for_backward: RefCell<Tensor<D>>,
}

impl<D, Input, Target> MeanSquaredError<D, Input, Target>
where
    D: Dimension,
    Input: Function<Dim = D>,
    Target: Function<Dim = D>,
{
    pub fn new(input: &Input, target: &Target) -> Self {
        assert_eq!(input.data().shape(), target.data().shape());

        Self {
            data: Rc::new(RefCell::new(arr0(0.0))),
            input: input.clone(),
            target: target.clone(),
            gradient: Rc::new(RefCell::new(Tensor::zeros(input.data().raw_dim()))),
            buffer_for_backward: RefCell::new(Tensor::zeros(input.data().raw_dim())),
        }
    }
}

impl<D, Input, Target> Function for MeanSquaredError<D, Input, Target>
where
    D: Dimension,
    Input: Function<Dim = D, GradDim = D>,
    Target: Function<Dim = D, GradDim = D>,
{
    type Dim = Ix0;
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
        self.target.forward();

        let squared_sum = Zip::from(&*self.input.data())
            .and(&*self.target.data())
            .fold(0.0, |sum, &input, &target| sum + (input - target).powi(2));
        *self.data.borrow_mut() = arr0(squared_sum / self.input.data().len() as f32 / 2.0);
    }

    fn backward(&self) {
        let diff = &*self.input.data() - &*self.target.data();
        self.input.update_gradient(&diff);
        // Omit updating `self.target`'s gradient because there is no need to diffrentiate w.r.t
        // target data.

        self.input.backward();
        self.target.backward();
    }
}

#[cfg(test)]
mod tests {
    use crate::{assert_rel_eq_arr2, grad::Variable};

    use super::*;

    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn compute_mse() {
        let y_pred = Variable::new(arr2(&[[1.0, 0.1], [-1.0, -1.5]])).requires_grad();
        let y_test = Variable::new(arr2(&[[1.0, 1.0], [3.0, -2.0]])).requires_grad();
        let z = mse(&y_pred, &y_test);
        z.forward();
        assert_rel_eq_arr2!(arr0(2.1325), z.data().clone());

        z.init_gradient();
        z.backward();
        assert_rel_eq_arr2!(arr2(&[[0.0, -0.9], [-4.0, 0.5]]), y_pred.gradient().clone());
    }
}
