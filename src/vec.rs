use std::ops::{Index, IndexMut};

pub struct Matrix(Vec<Vec<f64>>);

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index.0][index.1]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index.0][index.1]
    }
}

impl Matrix {
    pub fn zero(size: usize) -> Matrix {
        Matrix(vec![vec![0.0; size]; size])
    }

    pub fn into_vec(self) -> Vec<Vec<f64>> {
        self.0
    }

    pub fn dot(&self, v: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; v.len()];
        for (i, row) in self.0.iter().enumerate() {
            for (a, b) in row.iter().zip(v.iter()) {
                output[i] += *a * *b;
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compare_output, load_array_data, load_matrix_data};

    #[test]
    fn dot() {
        const SIZE: usize = 16;
        let mut mat = Matrix::zero(SIZE);
        load_matrix_data(SIZE, &mut mat);
        let mut arr = vec![vec![0.0; SIZE]; 100];
        load_array_data(SIZE, &mut arr);
        let mut outputs = Vec::with_capacity(arr.len());
        for a in arr {
            outputs.push(mat.dot(&a));
        }
        compare_output(SIZE, &outputs);
    }
}
