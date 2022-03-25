use std::{collections::HashMap, hash::Hash};

use ndarray::Array2;

pub mod activation;
pub mod data;
pub mod grad;
pub mod layer;
pub mod loss;
pub mod model;
pub mod network;
pub mod optimizer;

#[macro_export]
macro_rules! assert_rel_eq_arr1 {
    ($actual:expr, $expected:expr) => {
        assert_eq!($actual.shape(), $expected.shape());
        ndarray::Zip::from(&$actual)
            .and(&$expected)
            .for_each(|v, w| {
                assert_relative_eq!(v, w);
            });
    };
}

#[macro_export]
macro_rules! assert_rel_eq_arr2 {
    ($actual:expr, $expected:expr) => {
        assert_eq!($actual.shape(), $expected.shape());
        ndarray::Zip::from(&$actual)
            .and(&$expected)
            .for_each(|v, w| {
                assert_relative_eq!(v, w);
            });
    };
}

pub struct OneHotEncoder<Label>
where
    Label: Hash + Eq,
{
    label_to_id: HashMap<Label, usize>,
}

impl<Label> OneHotEncoder<Label>
where
    Label: Hash + Eq,
{
    pub fn new(label_kinds: Vec<Label>) -> Self {
        let label_to_id = label_kinds
            .into_iter()
            .enumerate()
            .map(|(id, label)| (label, id))
            .collect();
        Self { label_to_id }
    }

    fn encode_label(&self, label: &Label) -> Vec<f32> {
        let id = self.label_to_id.get(&label).expect("Unknown label");
        let mut one_hot = vec![0.0; self.label_to_id.len()];
        one_hot[*id] = 1.0;
        one_hot
    }

    pub fn encode(&self, labels: &[Label]) -> Array2<f32> {
        let one_hot_vecs = labels
            .into_iter()
            .map(|label| self.encode_label(label))
            .flatten()
            .collect();
        Array2::from_shape_vec((labels.len(), self.label_to_id.len()), one_hot_vecs).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_rel_eq_arr2;

    use super::*;

    use approx::assert_relative_eq;
    use ndarray::arr2;

    #[test]
    fn encode_labels() {
        let label_kinds = vec!["A", "B", "C"].into_iter().map(String::from).collect();
        let encoder = OneHotEncoder::new(label_kinds);

        let labels = vec!["A", "A", "C", "B", "C"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let one_hot_vecs = encoder.encode(&labels);
        assert_rel_eq_arr2!(
            arr2(&[
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0,],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0,],
            ]),
            one_hot_vecs
        );
    }
}
