/// Compute accuracy of the predicted labels `y_pred` to the correct labels `y_true`.
pub fn accuracy<Label>(y_true: &[Label], y_pred: &[Label]) -> f32
where
    Label: Eq,
{
    let n_corrects = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(t, p)| t == p)
        .count();
    n_corrects as f32 / y_true.len() as f32
}

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

/// Calculate the number of tp, fp, fn, tn.
/// `confusion_matrix` is a confusion matrix of all the labels.
/// `label_index` is the index of the label in the whole labels.
fn confusion_matrix_for_one_label(
    confusion_matrix: &[Vec<usize>],
    label_index: usize,
) -> (usize, usize, usize, usize) {
    let mut true_pos = 0;
    let mut false_pos = 0;
    let mut false_neg = 0;
    let mut true_neg = 0;

    for (i, row) in confusion_matrix.iter().enumerate() {
        for j in 0..row.len() {
            let count = confusion_matrix[i][j];
            if i == label_index && j == label_index {
                true_pos += count;
            } else if i == label_index {
                false_neg += count;
            } else if j == label_index {
                false_pos += count;
            } else {
                true_neg += count;
            }
        }
    }

    (true_pos, false_pos, false_neg, true_neg)
}

fn precision_for_one_label(true_pos: usize, false_pos: usize) -> f32 {
    let precision = true_pos as f32 / (true_pos + false_pos) as f32;
    if precision.is_nan() {
        0.0
    } else {
        precision
    }
}

fn recall_for_one_label(true_pos: usize, false_neg: usize) -> f32 {
    let recall = true_pos as f32 / (true_pos + false_neg) as f32;
    if recall.is_nan() {
        0.0
    } else {
        recall
    }
}

fn f1_for_one_label(precision: f32, recall: f32) -> f32 {
    let f1 = 2.0 * precision * recall / (precision + recall);
    if f1.is_nan() {
        0.0
    } else {
        f1
    }
}

pub fn precision_recall_fscore<Label>(
    y_true: &[Label],
    y_pred: &[Label],
    label_kinds: &[Label],
) -> (f32, f32, f32)
where
    Label: Eq + Clone,
{
    let confusion_matrix = confusion_matrix(y_true, y_pred, label_kinds);
    // Calculate macro average.
    let (precision_sum, recall_sum, f1_sum) = (0..label_kinds.len())
        .map(|label_index| {
            let (true_pos, false_pos, false_neg, _) =
                confusion_matrix_for_one_label(&confusion_matrix, label_index);
            let precision = precision_for_one_label(true_pos, false_pos);
            let recall = recall_for_one_label(true_pos, false_neg);
            let f1 = f1_for_one_label(precision, recall);
            (precision, recall, f1)
        })
        .fold((0.0, 0.0, 0.0), |(precision, recall, f1), (p, r, f)| {
            (precision + p, recall + r, f1 + f)
        });
    (
        precision_sum / label_kinds.len() as f32,
        recall_sum / label_kinds.len() as f32,
        f1_sum / label_kinds.len() as f32,
    )
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_accuracy() {
        let y_true = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let y_pred = vec![0, 0, 0, 1, 0, 1, 1, 2, 0, 1, 1, 2];
        assert_relative_eq!(0.5, accuracy(&y_true, &y_pred))
    }

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

    #[test]
    fn test_precision_recall() {
        let y_true = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let y_pred = vec![0, 0, 0, 1, 0, 1, 1, 2, 0, 1, 1, 2];
        let (precision, recall, f1) = precision_recall_fscore(&y_true, &y_pred, &[0, 1, 2]);
        assert_relative_eq!(0.5, precision);
        assert_relative_eq!(0.5, recall);
        assert_relative_eq!(0.48148152, f1);
    }

    #[test]
    fn precision_is_0_with_no_prediction_to_label() {
        let y_true = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        // No label to predict 2.
        let y_pred = vec![0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1];
        let confusion_matrix = confusion_matrix(&y_true, &y_pred, &[0, 1, 2]);
        let (tp, fp, _, _) = confusion_matrix_for_one_label(&confusion_matrix, 2);
        assert_relative_eq!(0.0, precision_for_one_label(tp, fp));
    }

    #[test]
    fn recall_is_0_with_no_true_label() {
        let y_true = vec![1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2];
        let y_pred = vec![0, 0, 0, 1, 0, 1, 1, 2, 0, 1, 1, 2];
        let confusion_matrix = confusion_matrix(&y_true, &y_pred, &[0, 1, 2]);
        let (tp, _, f_n, _) = confusion_matrix_for_one_label(&confusion_matrix, 0);
        assert_relative_eq!(0.0, precision_for_one_label(tp, f_n));
    }
}
