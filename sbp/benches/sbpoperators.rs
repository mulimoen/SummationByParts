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

    group.finish();
}

criterion_group!(benches, performance_benchmark);
criterion_main!(benches);
