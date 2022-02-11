use ndarray::Array2;

pub trait Activation {
    fn compute(&self, x: &Array2<f64>) -> Array2<f64>;

    fn derivative(&self, x: &Array2<f64>) -> Array2<f64>;
}

pub struct Identity;

impl Activation for Identity {
    fn compute(&self, x: &Array2<f64>) -> Array2<f64> {
        x.clone()
    }

    fn derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(|_| 1.0)
    }
}

pub struct Sigmoid;

impl Sigmoid {
    fn compute_one(x: &f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl Activation for Sigmoid {
    fn compute(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(Sigmoid::compute_one)
    }

    fn derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(|v| {
            let w = Sigmoid::compute_one(v);
            w * (1.0 - w)
        })
    }
}

pub struct Relu;

impl Activation for Relu {
    fn compute(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(|&v| if v > 0.0 { v } else { 0.0 })
    }

    fn derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_rel_eq_arr2;

    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn sigmoid_compute() {
        let x = arr2(&[[-2.0, -1.0, 0.0, 1.0, 2.0]]);
        let actual = Sigmoid.compute(&x);
        let expected = arr2(&[[
            0.1192029220221175,
            0.2689414213699951,
            0.5000000000000000,
            0.7310585786300049,
            0.8807970779778823,
        ]]);
        assert_rel_eq_arr2!(actual, expected);
    }

    #[test]
    fn sigmoid_derivative() {
        let x = arr2(&[[-2.0, -1.0, 0.0, 1.0, 2.0]]);
        let actual = Sigmoid.derivative(&x);
        let expected = arr2(&[[
            0.1049935854035065,
            0.1966119332414819,
            0.2500000000000000,
            0.1966119332414819,
            0.1049935854035066,
        ]]);
        assert_rel_eq_arr2!(actual, expected);
    }

    #[test]
    fn relu_compute() {
        let x = arr2(&[[-2.0, -1.0, 0.0, 1.0, 2.0]]);
        let actual = Relu.compute(&x);
        let expected = arr2(&[[0.0, 0.0, 0.0, 1.0, 2.0]]);
        assert_rel_eq_arr2!(actual, expected);
    }

    #[test]
    fn relu_derivative() {
        let x = arr2(&[[-2.0, -1.0, 0.0, 1.0, 2.0]]);
        let actual = Relu.derivative(&x);
        let expected = arr2(&[[0.0, 0.0, 0.0, 1.0, 1.0]]);
        assert_rel_eq_arr2!(actual, expected);
    }
}
