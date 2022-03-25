use std::{collections::HashMap, hash::Hash};

use ndarray::{Array2, Axis};

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

/// Encode labels to one-hot vectors and decode them.
pub struct OneHotEncoder<Label>
where
    Label: Hash + Eq + Clone,
{
    label_to_id: HashMap<Label, usize>,
    id_to_label: Vec<Label>,
}

impl<Label> OneHotEncoder<Label>
where
    Label: Hash + Eq + Clone,
{
    /// Record labels to convert.
    pub fn new(label_kinds: Vec<Label>) -> Self {
        let label_to_id = label_kinds
            .iter()
            .cloned()
            .enumerate()
            .map(|(id, label)| (label, id))
            .collect();
        Self {
            label_to_id,
            id_to_label: label_kinds,
        }
    }

    fn encode_label(&self, label: &Label) -> Vec<f32> {
        let id = self.label_to_id.get(label).expect("Unknown label");
        let mut one_hot = vec![0.0; self.label_to_id.len()];
        one_hot[*id] = 1.0;
        one_hot
    }

    /// Encode labels to one-hot vectors as 2D matrix whose shape is (n_data, n_features).
    /// Panics if an unknown label is passed.
    pub fn encode(&self, labels: &[Label]) -> Array2<f32> {
        let one_hot_vecs = labels
            .iter()
            .flat_map(|label| self.encode_label(label))
            .collect();
        Array2::from_shape_vec((labels.len(), self.label_to_id.len()), one_hot_vecs).unwrap()
    }

    /// Decode one-hot vectors to labels.
    /// Decoded label is determined by an argmax of each one-hot vector.
    pub fn decode(&self, one_hot_vecs: Array2<f32>) -> Vec<Label> {
        one_hot_vecs
            .lanes(Axis(1))
            .into_iter()
            .map(|one_hot| {
                // argmax of `one_hot`
                one_hot
                    .into_iter()
                    .enumerate()
                    .fold(
                        (0, 0.0f32),
                        |(max_index, max_elem), (index, &one_hot_elem)| {
                            if one_hot_elem > max_elem {
                                (index, one_hot_elem)
                            } else {
                                (max_index, max_elem)
                            }
                        },
                    )
                    .0
            })
            .map(|id| self.id_to_label[id].clone())
            .collect()
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

    #[test]
    fn decode_one_hot() {
        let label_kinds = vec!["A", "B", "C"].into_iter().map(String::from).collect();
        let encoder = OneHotEncoder::new(label_kinds);

        let one_hot_vecs = arr2(&[
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.05, 0.05, 0.9],
            [0.0, 1.0, 0.0],
            [0.1, 0.2, 0.7],
        ]);
        let labels = encoder.decode(one_hot_vecs);
        assert_eq!(
            vec!["A", "A", "C", "B", "C"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>(),
            labels
        );
    }
}
