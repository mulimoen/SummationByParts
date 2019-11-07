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

    let w = 35;
    let h = 26;
    let mut universe = Universe::new(w, h);

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
