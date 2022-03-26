use ndarray_rand::rand::{prelude::SliceRandom, thread_rng};

/// Split dataset into train and test data.
/// `test_ratio` is a ratio of the number of test data to the whole dataset.
pub fn train_test_split<X, Y>(x: Vec<X>, y: Vec<Y>, test_ratio: f32) -> (Vec<X>, Vec<Y>, Vec<X>, Vec<Y>)
where
    X: Clone,
    Y: Clone,
{
    assert_eq!(x.len(), y.len());

    let mut rng = thread_rng();
    let n_trains = (x.len() as f32 * (1.0 - test_ratio)) as usize;

    let mut zipped = x.into_iter().zip(y.into_iter()).collect::<Vec<_>>();
    zipped.shuffle(&mut rng);
    let (mut x_train, mut y_train): (Vec<_>, Vec<_>) = zipped.into_iter().unzip();

    let x_test = x_train.split_off(n_trains);
    let y_test = y_train.split_off(n_trains);

    (x_train, y_train, x_test, y_test)
}
