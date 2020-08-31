use criterion::{criterion_group, criterion_main, Criterion};
use sbp::operators::{self, SbpOperator2d};
use sbp::Float;

fn performance_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("SBPoperators");
    group.sample_size(25);

    let w = 64;
    let h = 64;

    let x = ndarray::Array2::from_shape_fn((w, h), |(j, i)| (j * h + i) as Float);

    group.bench_function("upwind4 diffxi", |b| {
        let mut res = x.clone();
        b.iter(|| operators::Upwind4.diffxi(x.view(), res.view_mut()))
    });
    group.bench_function("upwind9 diffxi", |b| {
        let mut res = x.clone();
        b.iter(|| operators::Upwind9.diffxi(x.view(), res.view_mut()))
    });
    group.bench_function("trad4 diffxi", |b| {
        let mut res = x.clone();
        b.iter(|| operators::SBP4.diffxi(x.view(), res.view_mut()))
    });
    group.bench_function("trad8 diffxi", |b| {
        let mut res = x.clone();
        b.iter(|| operators::SBP8.diffxi(x.view(), res.view_mut()))
    });

    group.bench_function("upwind4 diffeta", |b| {
        let mut res = x.clone();
        b.iter(|| operators::Upwind4.diffeta(x.view(), res.view_mut()))
    });
    group.bench_function("upwind9 diffeta", |b| {
        let mut res = x.clone();
        b.iter(|| operators::Upwind9.diffeta(x.view(), res.view_mut()))
    });
    group.bench_function("trad4 diffeta", |b| {
        let mut res = x.clone();
        b.iter(|| operators::SBP4.diffeta(x.view(), res.view_mut()))
    });
    group.bench_function("trad8 diffeta", |b| {
        let mut res = x.clone();
        b.iter(|| operators::SBP8.diffeta(x.view(), res.view_mut()))
    });

    #[cfg(feature = "sparse")]
    {
        use sbp::utils::kronecker_product;

        fn make_operators(
            op: impl SbpOperator2d,
            nx: usize,
            ny: usize,
        ) -> (sprs::CsMat<Float>, sprs::CsMat<Float>) {
            let dx = op.op_xi().diff_matrix(nx);
            let dx = kronecker_product(sprs::CsMat::eye(ny).view(), dx.view());

            let dy = op.op_eta().diff_matrix(ny);
            let dy = kronecker_product(dy.view(), sprs::CsMat::eye(nx).view());

            (dy, dx)
        }

        {
            let (dy, dx) = make_operators(operators::Upwind4, w, h);
            group.bench_function("upwind4 diffxi matrix", |b| {
                let mut res = ndarray::Array2::zeros(x.raw_dim());
                b.iter(|| {
                    sprs::prod::mul_acc_mat_vec_csr(
                        dx.view(),
                        x.as_slice().unwrap(),
                        res.as_slice_mut().unwrap(),
                    )
                })
            });
            group.bench_function("upwind4 diffeta matrix", |b| {
                let mut res = ndarray::Array2::zeros(x.raw_dim());
                b.iter(|| {
                    sprs::prod::mul_acc_mat_vec_csr(
                        dy.view(),
                        x.as_slice().unwrap(),
                        res.as_slice_mut().unwrap(),
                    )
                })
            });
        }
        {
            let (dy, dx) = make_operators(operators::Upwind9, w, h);
            group.bench_function("upwind9 diffxi matrix", |b| {
                let mut res = ndarray::Array2::zeros(x.raw_dim());
                b.iter(|| {
                    sprs::prod::mul_acc_mat_vec_csr(
                        dx.view(),
                        x.as_slice().unwrap(),
                        res.as_slice_mut().unwrap(),
                    )
                })
            });
            group.bench_function("upwind9 diffeta matrix", |b| {
                let mut res = ndarray::Array2::zeros(x.raw_dim());
                b.iter(|| {
                    sprs::prod::mul_acc_mat_vec_csr(
                        dy.view(),
                        x.as_slice().unwrap(),
                        res.as_slice_mut().unwrap(),
                    )
                })
            });
        }
    }

    group.finish();
}

criterion_group!(benches, performance_benchmark);
criterion_main!(benches);
