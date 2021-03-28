use criterion::{criterion_group, criterion_main, Criterion};

use naive_gemv_bench::{load_array_data, load_matrix_data, new_array1, new_array2, vec::Matrix};

const SAMPLES: usize = 100;

fn ndarray_128(c: &mut Criterion) {
    let m = new_array2(128);
    let arr = new_array1(128, SAMPLES);
    c.bench_function("ndarray_128", |b| {
        b.iter(|| {
            for a in arr.iter() {
                let _ = m.dot(a);
            }
        });
    });
}

fn vec_128(c: &mut Criterion) {
    let m = {
        let mut m = Matrix::zero(128);
        load_matrix_data(128, &mut m);
        m
    };
    let arr = {
        let mut arr = vec![vec![0.0; 128]; SAMPLES];
        load_array_data(128, &mut arr);
        arr
    };
    c.bench_function("vec_128", |b| {
        b.iter(|| {
            for a in arr.iter() {
                let _ = m.dot(a);
            }
        });
    });
}

criterion_group!(benches, ndarray_128, vec_128);
criterion_main!(benches);
