pub mod activation;
pub mod layer;
pub mod loss;
pub mod network;

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
