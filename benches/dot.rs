use criterion::{criterion_group, criterion_main, Criterion};

use naive_gemv_bench::{nalg, new_array1, new_array2, vec};

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

fn nalgebra_128(c: &mut Criterion) {
    let m = nalg::new_dynamic_matrix(128);
    let arr = nalg::new_dynamic_vector(128, SAMPLES);
    c.bench_function("nalgebra_128", |b| {
        b.iter(|| {
            for a in arr.iter() {
                let _ = &m * a;
            }
        });
    });
}

fn nalgebra_mul_to_128(c: &mut Criterion) {
    use nalgebra::DVector;

    let m = nalg::new_dynamic_matrix(128);
    let arr = nalg::new_dynamic_vector(128, SAMPLES);
    let mut output = DVector::zeros(128);
    c.bench_function("nalgebra_mul_to_128", |b| {
        b.iter(|| {
            for a in arr.iter() {
                m.mul_to(&a, &mut output);
            }
        });
    });
}

fn vec_128(c: &mut Criterion) {
    let m = vec::new_matrix(128);
    let arr = vec::new_array(128, SAMPLES);
    c.bench_function("vec_128", |b| {
        b.iter(|| {
            for a in arr.iter() {
                let _ = m.dot(a);
            }
        });
    });
}

criterion_group!(
    benches,
    ndarray_128,
    nalgebra_128,
    nalgebra_mul_to_128,
    vec_128,
);
criterion_main!(benches);
