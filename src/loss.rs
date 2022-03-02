use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

pub trait LossCriterion {
    fn compute(input: ArrayView2<f64>, target: ArrayView2<f64>) -> Self;

    fn value(&self) -> f64;

    fn grad(&self) -> Array2<f64>;
}

/// Computes mean squared error over a batch and elements for each data.
pub struct MeanSquaredError {
    input: Array2<f64>,
    target: Array2<f64>,
    value: f64,
}

impl LossCriterion for MeanSquaredError {
    fn compute(input: ArrayView2<f64>, target: ArrayView2<f64>) -> Self {
        assert_eq!(input.len(), target.len());

        let n = input.len();
        let loss = Zip::from(&input)
            .and(&target)
            .fold(0.0, |loss, &input, &target| loss + (input - target).powi(2))
            / n as f64;

        Self {
            input: input.to_owned(),
            target: target.to_owned(),
            value: loss,
        }
    }

    fn value(&self) -> f64 {
        self.value
    }

    fn grad(&self) -> Array2<f64> {
        let n = self.input.len();
        (&self.input - &self.target) / n as f64
    }
}

fn softmax(x: ArrayView1<f64>) -> Array1<f64> {
    let max_element = x.iter().fold(f64::NAN, |v, &w| v.max(w));
    let exp_each = x.map(|v| (v - max_element).exp());
    let exp_sum = exp_each.sum();
    exp_each / exp_sum
}

pub struct SoftmaxCrossEntropy {
    input: Array2<f64>,
    target: Array2<f64>,
    value: f64,
}

impl LossCriterion for SoftmaxCrossEntropy {
    fn compute(input: ArrayView2<f64>, target: ArrayView2<f64>) -> Self {
        let batch_size = input.nrows();
        let loss =
            Zip::from(input.rows())
                .and(target.rows())
                .fold(0.0, |batch_loss, input, target| {
                    let softmax = softmax(input.view());
                    batch_loss
                        + Zip::from(&target)
                            .and(&softmax)
                            .fold(0.0, |loss, t, y| loss + t * y.log2())
                })
                / (batch_size as f64 * -1.0);

        Self {
            input: input.to_owned(),
            target: target.to_owned(),
            value: loss,
        }
    }

    fn value(&self) -> f64 {
        self.value
    }

    fn grad(&self) -> Array2<f64> {
        let batch_size = self.input.nrows();
        (&self.input - &self.target) / batch_size as f64
    }
}

#[cfg(test)]
mod tests {
    use crate::{assert_rel_eq_arr1, assert_rel_eq_arr2};

    use super::*;

    use approx::assert_relative_eq;
    use ndarray::{arr1, arr2};

    fn prepare_mse() -> MeanSquaredError {
        let input = arr2(&[[1.0, 0.5, -0.1], [0.5, 0.2, 1.0]]);
        let target = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        MeanSquaredError::compute(input.view(), target.view())
    }

    #[test]
    fn compute_mse() {
        let loss = prepare_mse();
        assert_relative_eq!(0.35833333333333334, loss.value());
    }

    #[test]
    fn grad_mse() {
        let loss = prepare_mse();
        assert_rel_eq_arr2!(
            loss.grad(),
            arr2(&[
                [0.0, 0.0833333333333333, -0.0166666666666667],
                [0.0833333333333333, -0.1333333333333333, 0.1666666666666667]
            ])
        );
    }

    #[test]
    fn compute_softmax() {
        let x = arr1(&[1.0, 0.5, -0.1, 0.5, 0.2, 3.0]);
        assert_rel_eq_arr1!(
            arr1(&[
                0.0962990589663384,
                0.058408331764559,
                0.0320551721172303,
                0.058408331764559,
                0.0432699564108081,
                0.7115591489765052
            ]),
            softmax(x.view())
        );
    }

    fn prepare_softmax_cross_entropy() -> SoftmaxCrossEntropy {
        let input = arr2(&[[1.0, 0.5, -0.1], [0.5, 0.2, 3.0]]);
        let target = arr2(&[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        SoftmaxCrossEntropy::compute(input.view(), target.view())
    }

    #[test]
    fn compute_softmax_cross_entropy() {
        let loss = prepare_softmax_cross_entropy();
        assert_relative_eq!(2.3775211159596603, loss.value());
    }

    #[test]
    fn softmax_cross_entropy_grad() {
        let loss = prepare_softmax_cross_entropy();
        assert_rel_eq_arr2!(loss.grad(), arr2(&[[0.0, 0.25, -0.05], [-0.25, 0.1, 1.5]]));
    }
}
