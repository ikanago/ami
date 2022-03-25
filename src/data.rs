use std::vec;

use ndarray::{Axis, RemoveAxis};

use crate::grad::{Tensor, TensorView};

pub trait Sampler {
    fn sample(&mut self) -> Vec<usize>;
}

pub struct SequentialSampler {
    size: usize,
}

impl SequentialSampler {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Sampler for SequentialSampler {
    fn sample(&mut self) -> Vec<usize> {
        (0..self.size).into_iter().collect()
    }
}

pub struct Batch<'a, D1, D2>
where
    D1: RemoveAxis,
    D2: RemoveAxis,
{
    indices: vec::IntoIter<usize>,
    batch_size: usize,
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
            input,
            target,
        }
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

        if indices.is_empty() {
            None
        } else {
            Some((
                self.input.select(Axis(0), &indices),
                self.target.select(Axis(0), &indices),
            ))
        }
    }
}

pub struct DataLoader<S, D1, D2>
where
    S: Sampler,
    D1: RemoveAxis,
    D2: RemoveAxis,
{
    sampler: S,
    input: Tensor<D1>,
    target: Tensor<D2>,
}

impl<S, D1, D2> DataLoader<S, D1, D2>
where
    S: Sampler,
    D1: RemoveAxis,
    D2: RemoveAxis,
{
    pub fn new(sampler: S, input: Tensor<D1>, target: Tensor<D2>) -> Self {
        Self {
            sampler,
            input,
            target,
        }
    }

    pub fn batch<'a>(&'a mut self, batch_size: usize) -> Batch<'a, D1, D2> {
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

    #[test]
    fn batch_sequential() {
        let input = Array::linspace(0.0, 3.0, 4);
        let target = Array::linspace(4.0, 7.0, 4);
        let mut batch = Batch::new(
            SequentialSampler::new(input.len()).sample(),
            2,
            input.view(),
            target.view(),
        );
        assert_eq!(Some((arr1(&[0.0, 1.0]), arr1(&[4.0, 5.0]))), batch.next());
        assert_eq!(Some((arr1(&[2.0, 3.0]), arr1(&[6.0, 7.0]))), batch.next());
        assert_eq!(None, batch.next());
    }

    #[test]
    fn batch_sequential_keep_remaining() {
        let input = Array::linspace(0.0, 4.0, 5);
        let target = Array::linspace(5.0, 9.0, 5);
        let mut batch = Batch::new(
            SequentialSampler::new(input.len()).sample(),
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
