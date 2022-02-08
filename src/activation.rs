use ndarray::Array1;

pub struct Sigmoid;

impl Sigmoid {
    pub fn forward(self, x: Array1<f64>) -> Array1<f64> {
        x.map(|v| 1.0 / (1.0 + (-v).exp()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn sigmoid_forward() {
        let sigmoid = Sigmoid;
        let x = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let actual = sigmoid.forward(x);
        let expected = arr1(&[
            0.1192029220221175,
            0.2689414213699951,
            0.5000000000000000,
            0.7310585786300049,
            0.8807970779778823,
        ]);
        for i in 0..actual.len() {
            assert_relative_eq!(actual[i], expected[i]);
        }
    }
}
