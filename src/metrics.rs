/// Construct confusion matrix from `y_true` and `y_pred`.
/// An item in i-th row and j-th column is the number of predicted j-th label where a true label is
/// i-th one.
/// For correct result, the items in `label_kinds` must be unique.
pub fn confusion_matrix<Label>(
    y_true: &[Label],
    y_pred: &[Label],
    label_kinds: &[Label],
) -> Vec<Vec<usize>>
where
    Label: Eq + Clone,
{
    label_kinds
        .iter()
        .map(|true_label| {
            label_kinds
                .iter()
                .map(|pred_label| {
                    y_true
                        .iter()
                        .zip(y_pred.iter())
                        .filter(|(t, p)| &(*t).clone() == true_label && &(*p).clone() == pred_label)
                        .count()
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix() {
        let y_true = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let y_pred = vec![0, 0, 0, 1, 0, 1, 1, 2, 0, 1, 1, 2];
        assert_eq!(
            vec![vec![3, 1, 0], vec![1, 2, 1], vec![1, 2, 1]],
            confusion_matrix(&y_true, &y_pred, &[0, 1, 2])
        );
    }

    #[test]
    fn test_confusion_matrix_on_string_label() {
        let y_true = vec![
            "ant", "ant", "ant", "cat", "cat", "cat", "dog", "dog", "dog",
        ];
        let y_pred = vec![
            "ant", "ant", "cat", "cat", "cat", "cat", "ant", "cat", "dog",
        ];
        assert_eq!(
            vec![vec![2, 1, 0], vec![0, 3, 0], vec![1, 1, 1]],
            confusion_matrix(&y_true, &y_pred, &["ant", "cat", "dog"])
        );
    }
}
