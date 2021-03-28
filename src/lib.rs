use std::ops::{Index, IndexMut};

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_pcg::Mcg128Xsl64;

pub fn load_matrix_data<M>(size: usize, m: &mut M)
where
    M: IndexMut<(usize, usize), Output = f64> + Sized,
{
    // Deterministic random number generator
    let mut rng = Mcg128Xsl64::new(1);
    for i in 0..size {
        for j in 0..size {
            m[(i, j)] = rng.gen_range(-3.0..3.0);
        }
    }
}

pub fn load_array_data<M>(size: usize, arr: &mut [M])
where
    M: IndexMut<usize, Output = f64> + Sized,
{
    // Deterministic random number generator
    let mut rng = Mcg128Xsl64::new(11235);
    for a in arr {
        for i in 0..size {
            a[i] = rng.gen_range(-3.0..3.0);
        }
    }
}

pub fn compare_output<M>(size: usize, samples: &[M])
where
    M: Index<usize, Output = f64> + Sized,
{
    let m = {
        let mut m = Array2::zeros([size, size]);
        load_matrix_data(size, &mut m);
        m
    };
    let arr = {
        let mut arr = Vec::with_capacity(samples.len());
        for _ in 0..samples.len() {
            arr.push(Array1::zeros([size]));
        }
        load_array_data(size, &mut arr);
        arr
    };
    for (a, actual) in arr.iter().zip(samples) {
        let expect = m.dot(a);
        for i in 0..size {
            assert!((expect[i] - actual[i]).abs() < 1e-8);
        }
    }
}
