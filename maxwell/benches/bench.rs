use criterion::{black_box, criterion_group, criterion_main, Criterion};
use maxwell::System;
use sbp::operators::{SbpOperator2d, Upwind4, UpwindOperator2d, SBP4};
use sbp::Float;

fn advance_system<SBP: SbpOperator2d>(universe: &mut System<SBP>, n: usize) {
    for _ in 0..n {
        universe.advance(0.01);
    }
}

fn advance_system_upwind<UO: UpwindOperator2d>(universe: &mut System<UO>, n: usize) {
    for _ in 0..n {
        universe.advance_upwind(0.01);
    }
}

#[cfg(feature = "sparse")]
fn advance_system_matrix<SBP: SbpOperator2d>(universe: &mut System<SBP>, n: usize) {
    for _ in 0..n {
        universe.advance_sparse(0.01);
    }
}

#[cfg(feature = "sparse")]
fn advance_system_matrix_implicit<SBP: SbpOperator2d>(universe: &mut System<SBP>, n: usize) {
    for _ in 0..n {
        universe.advance_implicit();
    }
}

fn performance_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("MaxwellSystem");
    group.sample_size(25);

    let w = 40;
    let h = 26;
    let x = ndarray::Array2::from_shape_fn((h, w), |(_, i)| i as Float / (w - 1) as Float);
    let y = ndarray::Array2::from_shape_fn((h, w), |(j, _)| j as Float / (h - 1) as Float);

    let mut universe = System::new(x.clone(), y.clone(), Upwind4);
    group.bench_function("advance", |b| {
        b.iter(|| {
            universe.set_gaussian(0.5, 0.5);
            advance_system(&mut universe, black_box(20))
        })
    });

    let mut universe = System::new(x.clone(), y.clone(), Upwind4);
    group.bench_function("advance_upwind", |b| {
        b.iter(|| {
            universe.set_gaussian(0.5, 0.5);
            advance_system_upwind(&mut universe, black_box(20))
        })
    });

    let mut universe = System::new(x.clone(), y.clone(), SBP4);
    group.bench_function("advance_trad4", |b| {
        b.iter(|| {
            universe.set_gaussian(0.5, 0.5);
            advance_system(&mut universe, black_box(20))
        })
    });

    #[cfg(feature = "sparse")]
    {
        let mut universe = System::new(x.clone(), y.clone(), Upwind4);
        group.bench_function("advance_upwind4_as_matrix", |b| {
            b.iter(|| {
                universe.set_gaussian(0.5, 0.5);
                advance_system_matrix(&mut universe, black_box(20))
            })
        });
        let mut universe = System::new(x.clone(), y.clone(), sbp::operators::Upwind9);
        group.bench_function("advance_upwind9_as_matrix", |b| {
            b.iter(|| {
                universe.set_gaussian(0.5, 0.5);
                advance_system_matrix(&mut universe, black_box(20))
            })
        });
        let mut universe = System::new(x.clone(), y.clone(), Upwind4);
        group.bench_function("advance_upwind9_as_matrix_implicit", |b| {
            b.iter(|| {
                universe.set_gaussian(0.5, 0.5);
                advance_system_matrix_implicit(&mut universe, black_box(20))
            })
        });
    }

    group.finish();
}

#[cfg(feature = "sparse")]
fn sparse_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("MaxwellSystem");
    group.sample_size(25);
    let w = 40;
    let h = 26;
    group.bench_function("create_rhs_trad4", |b| {
        b.iter(|| {
            let _matrix = maxwell::sparse::rhs_matrix(&SBP4, w, h);
        })
    });
    group.bench_function("create_rhs_upwind4", |b| {
        b.iter(|| {
            let _matrix = maxwell::sparse::rhs_matrix(&sbp::operators::Upwind4, w, h);
        })
    });
    group.bench_function("create_rhs_upwind9", |b| {
        b.iter(|| {
            let _matrix = maxwell::sparse::rhs_matrix(&sbp::operators::Upwind9, w, h);
        })
    });

    group.bench_function("create_rhs_upwind_upwind4", |b| {
        b.iter(|| {
            let _matrix =
                maxwell::sparse::rhs_matrix_with_upwind_dissipation(&sbp::operators::Upwind4, w, h);
        })
    });

    group.bench_function("create_rhs_upwind_upwind9", |b| {
        b.iter(|| {
            let _matrix =
                maxwell::sparse::rhs_matrix_with_upwind_dissipation(&sbp::operators::Upwind9, w, h);
        })
    });
}

#[cfg(feature = "sparse")]
criterion_group!(sparse_create, sparse_creation);
criterion_group!(benches, performance_benchmark);

#[cfg(feature = "sparse")]
criterion_main!(benches, sparse_create);
#[cfg(not(feature = "sparse"))]
criterion_main!(benches);
