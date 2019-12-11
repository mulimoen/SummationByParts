use criterion::{black_box, criterion_group, criterion_main, Criterion};
use maxwell::Universe;

fn advance_system(universe: &mut Universe, n: usize) {
    for _ in 0..n {
        universe.advance(0.01);
    }
}

fn performance_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("System");
    group.sample_size(15);

    let w = 40;
    let h = 26;
    let x = ndarray::Array2::from_shape_fn((h, w), |(_, i)| i as f32 / (w - 1) as f32);
    let y = ndarray::Array2::from_shape_fn((h, w), |(j, _)| j as f32 / (h - 1) as f32);
    let mut universe = Universe::new(
        w as u32,
        h as u32,
        x.as_slice().unwrap(),
        y.as_slice().unwrap(),
    );

    group.bench_function("advance", |b| {
        b.iter(|| {
            universe.init(0.0, 0.0);
            advance_system(&mut universe, black_box(20))
        })
    });

    group.finish();
}

criterion_group!(benches, performance_benchmark);
criterion_main!(benches);
