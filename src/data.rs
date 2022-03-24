use std::ops::Range;

pub trait Sampler: Iterator {}

pub struct SequentialSampler {
    iter: Range<usize>,
}

impl SequentialSampler {
    pub fn new(size: usize) -> Self {
        Self { iter: (0..size) }
    }
}

impl Iterator for SequentialSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl Sampler for SequentialSampler {}

pub struct BatchSampler<S: Sampler> {
    individual_sampler: S,
    batch_size: usize,
}

impl<S: Sampler> BatchSampler<S> {
    pub fn new(sampler: S, batch_size: usize) -> Self {
        Self {
            individual_sampler: sampler,
            batch_size,
        }
    }
}

impl<S> Iterator for BatchSampler<S>
where
    S: Sampler<Item = usize>,
{
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut indices = Vec::new();
        for _ in 0..self.batch_size {
            match self.individual_sampler.next() {
                Some(index) => {
                    indices.push(index);
                }
                None => break,
            }
        }

        if indices.len() == 0 {
            None
        } else {
            Some(indices)
        }
    }
}

impl<S> Sampler for BatchSampler<S> where S: Sampler<Item = usize> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_sequential() {
        let mut batch_sampler = BatchSampler::new(SequentialSampler::new(6), 2);
        assert_eq!(Some(vec![0, 1]), batch_sampler.next());
        assert_eq!(Some(vec![2, 3]), batch_sampler.next());
        assert_eq!(Some(vec![4, 5]), batch_sampler.next());
        assert_eq!(None, batch_sampler.next());
    }

    #[test]
    fn batch_sequential_keep_remaining() {
        let mut batch_sampler = BatchSampler::new(SequentialSampler::new(5), 2);
        assert_eq!(Some(vec![0, 1]), batch_sampler.next());
        assert_eq!(Some(vec![2, 3]), batch_sampler.next());
        assert_eq!(Some(vec![4]), batch_sampler.next());
        assert_eq!(None, batch_sampler.next());
    }
}
