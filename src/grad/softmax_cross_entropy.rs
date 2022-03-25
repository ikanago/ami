use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{arr0, Axis, Dimension, Ix0, Zip};

use super::{Function, Tensor};

pub fn softmax_cross_entropy<D, Input, Target>(
    input: &Input,
    target: &Target,
) -> SoftmaxCrossEntropy<D, Input, Target>
where
    D: Dimension,
    Input: Function<Dim = D>,
    Target: Function<Dim = D>,
{
    SoftmaxCrossEntropy::new(input, target)
}

#[derive(Clone)]
pub struct SoftmaxCrossEntropy<D, Input, Target>
where
    D: Dimension,
    Input: Function,
    Target: Function,
{
    data: Rc<RefCell<Tensor<Ix0>>>,
    input: Input,
    target: Target,
    gradient: Rc<RefCell<Tensor<D>>>,
}

impl<D, Input, Target> SoftmaxCrossEntropy<D, Input, Target>
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
        }
    }
}

impl<D, Input, Target> Function for SoftmaxCrossEntropy<D, Input, Target>
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

        let sum_loss = Zip::from(self.input.data().lanes(Axis(1)))
            .and(self.target.data().lanes(Axis(1)))
            .fold(0.0, |loss, input_lane, target_lane| {
                let max_in_input = input_lane.fold(std::f32::MIN, |acc, x| acc.max(*x));
                let exp_input = input_lane.map(|x| (x - max_in_input).exp());
                let exp_sum = exp_input.sum();

                loss + Zip::from(&exp_input)
                    .and(target_lane)
                    .fold(0.0, |acc, input, target| acc + target * (input / exp_sum).log2())
            });

        *self.data.borrow_mut() = arr0(- sum_loss / self.input.data().len_of(Axis(0)) as f32);
    }

    fn backward(&self) {
        let mut diff = &*self.input.data() - &*self.target.data();
        diff /= self.input.data().len_of(Axis(0)) as f32;
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
    fn compute_softmax_cross_entropy() {
        let y_pred = Variable::new(arr2(&[[0.9, 0.2, 0.3], [0.3, 0.4, 0.3], [0.1, 0.4, 0.6]]))
            .requires_grad();
        let target = Variable::new(arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]));
        let z = softmax_cross_entropy(&y_pred, &target);
        z.forward();
        assert_rel_eq_arr2!(arr0(1.2669747613017643), z.data().clone());

        z.init_gradient();
        z.backward();
        assert_rel_eq_arr2!(
            arr2(&[
                [-0.0333333333333333, 0.0666666666666667, 0.1],
                [0.1, -0.2, 0.1],
                [0.0333333333333333, 0.1333333333333333, -0.1333333333333333]
            ]),
            y_pred.gradient().clone()
        );
    }
}
