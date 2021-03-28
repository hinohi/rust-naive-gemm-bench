use nalgebra::{DMatrix, DVector};

use crate::{load_array_data, load_matrix_data};

pub fn new_dynamic_matrix(size: usize) -> DMatrix<f64> {
    let mut m = DMatrix::zeros(size, size);
    load_matrix_data(size, &mut m);
    m
}

pub fn new_dynamic_vector(size: usize, samples: usize) -> Vec<DVector<f64>> {
    let mut arr = Vec::with_capacity(samples);
    for _ in 0..samples {
        arr.push(DVector::zeros(size));
    }
    load_array_data(size, &mut arr);
    arr
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare_output;

    #[test]
    fn dot() {
        const SIZE: usize = 16;
        let m = new_dynamic_matrix(SIZE);
        let arr = new_dynamic_vector(SIZE, 10);
        let mut outputs = Vec::with_capacity(arr.len());
        for a in arr {
            outputs.push(&m * &a);
        }
        compare_output(SIZE, &outputs);
    }
}
