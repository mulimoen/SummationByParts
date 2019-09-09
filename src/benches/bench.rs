use criterion::{black_box, criterion_group, criterion_main, Criterion};
use maxwell::Universe;

fn simple_system(w: u32, h: u32) -> Universe {
    let mut universe = Universe::new(w, h);
    universe.init(0.0, 0.0);
    for _ in 0..100 {
        universe.advance(0.01);
    }
    universe
}

fn performance_benchmark(c: &mut Criterion) {
    c.bench_function("complete system", |b| {
        b.iter(|| simple_system(black_box(25), black_box(30)))
    });
}

criterion_group!(benches, performance_benchmark);
criterion_main!(benches);
