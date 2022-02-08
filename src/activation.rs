use ndarray::Array2;

pub struct Sigmoid;

impl Sigmoid {
    fn compute_one(x: &f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn compute(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(Sigmoid::compute_one)
    }

    pub fn derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        x.map(|v| {
            let w = Sigmoid::compute_one(v);
            w * (1.0 - w)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn sigmoid_compute() {
        let sigmoid = Sigmoid;
        let x = arr2(&[[-2.0, -1.0, 0.0, 1.0, 2.0]]);
        let actual = sigmoid.compute(&x);
        let expected = arr2(&[[
            0.1192029220221175,
            0.2689414213699951,
            0.5000000000000000,
            0.7310585786300049,
            0.8807970779778823,
        ]]);
        for i in 0..actual.len() {
            assert_relative_eq!(actual[[0, i]], expected[[0, i]]);
        }
    }

    #[test]
    fn sigmoid_derivative() {
        let sigmoid = Sigmoid;
        let x = arr2(&[[-2.0, -1.0, 0.0, 1.0, 2.0]]);
        let actual = sigmoid.derivative(&x);
        let expected = arr2(&[[
            0.1049935854035065,
            0.1966119332414819,
            0.2500000000000000,
            0.1966119332414819,
            0.1049935854035066,
        ]]);
        for i in 0..actual.len() {
            assert_relative_eq!(actual[[0, i]], expected[[0, i]]);
        }
    }
}
