use criterion::{black_box, criterion_group, criterion_main, Criterion};
use sbp::euler::System;
use sbp::operators::{SbpOperator, Upwind4, UpwindOperator, SBP4};

fn advance_system<SBP: SbpOperator>(universe: &mut System<SBP>, n: usize) {
    for _ in 0..n {
        universe.advance(1.0 / 40.0 * 0.2);
    }
}

fn advance_system_upwind<UO: UpwindOperator>(universe: &mut System<UO>, n: usize) {
    for _ in 0..n {
        universe.advance_upwind(1.0 / 40.0 * 0.2);
    }
}

fn performance_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("EulerSystem");
    group.sample_size(25);

    let w = 40;
    let h = 26;
    let x = ndarray::Array1::linspace(-10.0, 10.0, w);
    let x = x.broadcast((h, w)).unwrap();
    let y = ndarray::Array1::linspace(-10.0, 10.0, h);
    let y = y.broadcast((w, h)).unwrap().reversed_axes();

    let mut universe = System::new(x.into_owned(), y.into_owned(), Upwind4);
    group.bench_function("advance", |b| {
        b.iter(|| {
            universe.init_with_vortex(0.0, 0.0);
            advance_system(&mut universe, black_box(20))
        })
    });

    let mut universe = System::new(x.into_owned(), y.into_owned(), Upwind4);
    group.bench_function("advance_upwind", |b| {
        b.iter(|| {
            universe.init_with_vortex(0.0, 0.0);
            advance_system_upwind(&mut universe, black_box(20))
        })
    });

    let mut universe = System::new(x.into_owned(), y.into_owned(), SBP4);
    group.bench_function("advance_trad4", |b| {
        b.iter(|| {
            universe.init_with_vortex(0.0, 0.0);
            advance_system(&mut universe, black_box(20))
        })
    });

    group.finish();
}

criterion_group!(benches, performance_benchmark);
criterion_main!(benches);
