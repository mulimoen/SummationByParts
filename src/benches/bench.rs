use criterion::{black_box, criterion_group, criterion_main, Criterion};
use maxwell::operators::{SbpOperator, Upwind4, UpwindOperator, SBP4};
use maxwell::{EulerSystem, System};

fn advance_system<SBP: SbpOperator>(universe: &mut System<SBP>, n: usize) {
    for _ in 0..n {
        universe.advance(0.01);
    }
}

fn advance_system_upwind<UO: UpwindOperator>(universe: &mut System<UO>, n: usize) {
    for _ in 0..n {
        universe.advance_upwind(0.01);
    }
}

fn performance_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("System");
    group.sample_size(25);

    let w = 40;
    let h = 26;
    let x = ndarray::Array2::from_shape_fn((h, w), |(_, i)| i as f32 / (w - 1) as f32);
    let y = ndarray::Array2::from_shape_fn((h, w), |(j, _)| j as f32 / (h - 1) as f32);

    let mut universe = System::<Upwind4>::new(w, h, x.as_slice().unwrap(), y.as_slice().unwrap());
    group.bench_function("advance", |b| {
        b.iter(|| {
            universe.set_gaussian(0.5, 0.5);
            advance_system(&mut universe, black_box(20))
        })
    });

    let mut universe = System::<Upwind4>::new(w, h, x.as_slice().unwrap(), y.as_slice().unwrap());
    group.bench_function("advance_upwind", |b| {
        b.iter(|| {
            universe.set_gaussian(0.5, 0.5);
            advance_system_upwind(&mut universe, black_box(20))
        })
    });

    let mut universe = System::<SBP4>::new(w, h, x.as_slice().unwrap(), y.as_slice().unwrap());
    group.bench_function("advance_trad4", |b| {
        b.iter(|| {
            universe.set_gaussian(0.5, 0.5);
            advance_system(&mut universe, black_box(20))
        })
    });

    group.finish();
}

criterion_group!(benches, performance_benchmark);

fn advance_euler_system<SBP: SbpOperator>(universe: &mut EulerSystem<SBP>, n: usize) {
    for _ in 0..n {
        universe.advance(0.01);
    }
}

fn advance_euler_system_upwind<UO: UpwindOperator>(universe: &mut EulerSystem<UO>, n: usize) {
    for _ in 0..n {
        universe.advance_upwind(0.01);
    }
}

fn euler_performance_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("EulerSystem");
    group.sample_size(25);

    let w = 40;
    let h = 26;
    let x = ndarray::Array1::linspace(-10.0, 10.0, w);
    let x = x.broadcast((h, w)).unwrap();
    let y = ndarray::Array1::linspace(-10.0, 10.0, h);
    let y = y.broadcast((w, h)).unwrap().reversed_axes();

    let mut universe = EulerSystem::<Upwind4>::new(x.into_owned(), y.into_owned());
    group.bench_function("advance_euler", |b| {
        b.iter(|| {
            universe.vortex(0.0, 0.0);
            advance_euler_system(&mut universe, black_box(10))
        })
    });

    let mut universe = EulerSystem::<Upwind4>::new(x.into_owned(), y.into_owned());
    group.bench_function("advance_euler_upwind", |b| {
        b.iter(|| {
            universe.vortex(0.0, 0.0);
            advance_euler_system_upwind(&mut universe, black_box(10))
        })
    });

    let mut universe = EulerSystem::<SBP4>::new(x.into_owned(), y.into_owned());
    group.bench_function("advance_euler_trad4", |b| {
        b.iter(|| {
            universe.vortex(0.0, 0.0);
            advance_euler_system(&mut universe, black_box(10))
        })
    });

    group.finish();
}

criterion_group!(euler_benches, euler_performance_benchmark);
criterion_main!(benches, euler_benches);
