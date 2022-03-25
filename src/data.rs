use std::vec;

use ndarray::{Axis, RemoveAxis};
use ndarray_rand::rand::{prelude::ThreadRng, seq::index::sample, thread_rng};

use crate::grad::{Tensor, TensorView};

/// Sampler produces a vector of indices in a dataset.
pub enum Sampler {
    Sequential(usize),
    Random(usize, ThreadRng),
}

impl Sampler {
    pub fn sample(&mut self) -> Vec<usize> {
        match self {
            Self::Sequential(size) => (0..*size).into_iter().collect(),
            Self::Random(size, rng) => sample(rng, *size, *size).into_vec(),
        }
    }
}

/// Batch yields a minibatch each time `Iterator::next()` is called.
/// This struct is created in each epoch in a train phase.
pub struct Batch<'a, D1, D2>
where
    D1: RemoveAxis,
    D2: RemoveAxis,
{
    // `indices` should be an `Iterator` because we have to keep track of the current batch
    // indices.
    indices: vec::IntoIter<usize>,
    batch_size: usize,
    drop_last: bool,
    input: TensorView<'a, D1>,
    target: TensorView<'a, D2>,
}

impl<'a, D1, D2> Batch<'a, D1, D2>
where
    D1: RemoveAxis,
    D2: RemoveAxis,
{
    pub fn new(
        indices: Vec<usize>,
        batch_size: usize,
        input: TensorView<'a, D1>,
        target: TensorView<'a, D2>,
    ) -> Self {
        Self {
            indices: indices.into_iter(),
            batch_size,
            drop_last: false,
            input,
            target,
        }
    }

    /// If `drop_last` is true, discard last minibatch whose size is smaller than
    /// `self.batch_size`.
    pub fn drop_last(self, drop_last: bool) -> Self {
        Self { drop_last, ..self }
    }

    fn should_drop_last(&self, indices: &[usize]) -> bool {
        self.drop_last && indices.len() != self.batch_size
    }
}

impl<'a, D1, D2> Iterator for Batch<'a, D1, D2>
where
    D1: RemoveAxis,
    D2: RemoveAxis,
{
    type Item = (Tensor<D1>, Tensor<D2>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut indices = Vec::new();
        for _ in 0..self.batch_size {
            match self.indices.next() {
                Some(index) => {
                    indices.push(index);
                }
                None => break,
            }
        }

        if indices.is_empty() || self.should_drop_last(&indices) {
            None
        } else {
            Some((
                self.input.select(Axis(0), &indices),
                self.target.select(Axis(0), &indices),
            ))
        }
    }
}

/// DataLoader wraps a training data.
pub struct DataLoader<D1, D2>
where
    D1: RemoveAxis,
    D2: RemoveAxis,
{
    sampler: Sampler,
    input: Tensor<D1>,
    target: Tensor<D2>,
}

impl<D1, D2> DataLoader<D1, D2>
where
    D1: RemoveAxis,
    D2: RemoveAxis,
{
    pub fn new(input: Tensor<D1>, target: Tensor<D2>) -> Self {
        Self {
            sampler: Sampler::Sequential(input.len_of(Axis(0))),
            input,
            target,
        }
    }

    pub fn size(&self) -> usize {
        self.input.len_of(Axis(0))
    }

    /// If enabled, generate minibatches in a random order.
    pub fn shuffle(mut self) -> Self {
        self.sampler = Sampler::Random(self.size(), thread_rng());
        self
    }

    /// Create a minibatch generator. This is intended to be called each epoch.
    pub fn batch(&mut self, batch_size: usize) -> Batch<'_, D1, D2> {
        Batch::new(
            self.sampler.sample(),
            batch_size,
            self.input.view(),
            self.target.view(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{arr1, Array};
    use ndarray_rand::rand::thread_rng;

    #[test]
    fn random_sampler() {
        let size = 10;
        let sequential = Sampler::Sequential(size).sample();
        let mut random = Sampler::Random(size, thread_rng()).sample();
        assert_ne!(sequential, random);

        random.sort();
        assert_eq!(sequential, random);
    }

    #[test]
    fn batch_sequential() {
        let input = Array::linspace(0.0, 3.0, 4);
        let target = Array::linspace(4.0, 7.0, 4);
        let mut batch = Batch::new(
            Sampler::Sequential(input.len()).sample(),
            2,
            input.view(),
            target.view(),
        );
        assert_eq!(Some((arr1(&[0.0, 1.0]), arr1(&[4.0, 5.0]))), batch.next());
        assert_eq!(Some((arr1(&[2.0, 3.0]), arr1(&[6.0, 7.0]))), batch.next());
        assert_eq!(None, batch.next());
    }

    #[test]
    fn batch_sequential_drop_remaining() {
        let input = Array::linspace(0.0, 4.0, 5);
        let target = Array::linspace(5.0, 9.0, 5);
        let mut batch = Batch::new(
            Sampler::Sequential(input.len()).sample(),
            2,
            input.view(),
            target.view(),
        )
        .drop_last(true);
        assert_eq!(Some((arr1(&[0.0, 1.0]), arr1(&[5.0, 6.0]))), batch.next());
        assert_eq!(Some((arr1(&[2.0, 3.0]), arr1(&[7.0, 8.0]))), batch.next());
        assert_eq!(None, batch.next());
    }

    #[test]
    fn batch_sequential_keep_remaining() {
        let input = Array::linspace(0.0, 4.0, 5);
        let target = Array::linspace(5.0, 9.0, 5);
        let mut batch = Batch::new(
            Sampler::Sequential(input.len()).sample(),
            2,
            input.view(),
            target.view(),
        );
        assert_eq!(Some((arr1(&[0.0, 1.0]), arr1(&[5.0, 6.0]))), batch.next());
        assert_eq!(Some((arr1(&[2.0, 3.0]), arr1(&[7.0, 8.0]))), batch.next());
        assert_eq!(Some((arr1(&[4.0]), arr1(&[9.0]))), batch.next());
        assert_eq!(None, batch.next());
    }
}
