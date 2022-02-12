use ndarray::{Array, Array2, Axis};
use ndarray_rand::rand::{seq::SliceRandom, thread_rng};

// Create a matrix with (size, size) shape to shuffle `Array2`.
// The matrix only one `1.0` element for each row and column.
// Multiplied from left, the multiplied matrix is shuffled along `Axis(0)`
// and multiplied from right, the multiplied matrix is shuffled along `Axis(1)`.
fn create_shuffle_matrix(size: usize) -> Array2<f64> {
    let mut index = (0..size).into_iter().collect::<Vec<_>>();
    let mut rng = thread_rng();
    index.shuffle(&mut rng);

    let index = index
        .into_iter()
        .flat_map(|i| {
            let mut v = vec![0.0; size];
            v[i] = 1.0;
            v
        })
        .collect::<Vec<_>>();
    Array::from_shape_vec((size, size), index).unwrap()
}

pub fn train_test_split(
    x: Array2<f64>,
    y: Array2<f64>,
    train_ratio: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    assert!(x.len_of(Axis(0)) == y.len_of(Axis(0)));
    assert!(0.0 < train_ratio && train_ratio < 1.0);

    let dataset_size = x.len_of(Axis(0));
    let shuffle_matrix = create_shuffle_matrix(dataset_size);
    // TODO: use permuted_axes: https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.permuted_axes
    let x = shuffle_matrix.dot(&x);
    let y = shuffle_matrix.dot(&y);

    let train_size = ((dataset_size as f64) * train_ratio) as usize;
    let (x_train, x_test) = x.view().split_at(Axis(0), train_size);
    let (y_train, y_test) = y.view().split_at(Axis(0), train_size);
    (
        x_train.to_owned(),
        x_test.to_owned(),
        y_train.to_owned(),
        y_test.to_owned(),
    )
}
