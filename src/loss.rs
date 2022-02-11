use ndarray::{Array2, ArrayView2, Zip};

pub trait Loss {
    fn compute(input: ArrayView2<f64>, target: ArrayView2<f64>) -> Self;

    fn value(&self) -> f64;

    fn grad(&self) -> Array2<f64>;
}

pub struct MeanSquaredError {
    input: Array2<f64>,
    target: Array2<f64>,
    value: f64,
}

impl Loss for MeanSquaredError {
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
        let mut grad = &self.input - &self.target;
        grad /= n as f64;
        grad
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_rel_eq_arr2;

    use super::*;

    use approx::assert_relative_eq;
    use ndarray::arr2;

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
}
