use ndarray::{Array1, ArrayView1};
use plotters::prelude::*;
use sbp::{
    integrate::{integrate, Rk4},
    operators::{SbpOperator1d2, SBP4},
    Float,
};

fn main() {
    let nx: usize = 101;
    let x = Array1::from_shape_fn((nx,), |i| i as Float / (nx - 1) as Float);
    let v0 = x.map(|&x| (-(x - 0.5).powi(2) / 0.1).exp());
    dual_dirichlet(v0.view(), 1.0, 1.0);
    // Neumann boundary is introducing energy into the system
    neumann_dirichlet(v0.view(), -0.2, 1.0);
}

fn dual_dirichlet(v: ArrayView1<Float>, v0: Float, vn: Float) {
    let drawing_area = BitMapBackend::gif("dual_dirichlet.gif", (300, 300), 100)
        .unwrap()
        .into_drawing_area();
    let mut chart = ChartBuilder::on(&drawing_area)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..1.01, -1.0..2.0)
        .unwrap();

    let nx = v.len();
    let dt = 0.2 / nx.pow(2) as Float / 3.0;
    let x = Array1::from_shape_fn((nx,), |i| i as Float / (nx - 1) as Float);

    let op = SBP4;

    let mut k = [v.to_owned(), v.to_owned(), v.to_owned(), v.to_owned()];
    let rhs = move |fut: &mut Array1<Float>, prev: &Array1<Float>, _t: Float| {
        fut.fill(0.0);
        op.diff2(prev.view(), fut.view_mut());

        let h = 1.0 / (nx - 1) as Float;
        let tau = (1.0, -1.0);

        let d1 = op.d1();

        for (d, fut) in d1.iter().zip(fut.iter_mut()) {
            *fut += tau.0 / (h * h) * d * (prev[0] - v0);
        }
        for (d, fut) in d1.iter().zip(fut.iter_mut().rev().take(d1.len()).rev()) {
            *fut += tau.1 / (h * h) * d * (prev[nx - 1] - vn);
        }
    };

    let mut v1 = v.to_owned();
    let mut v2 = v.to_owned();
    for i in 0..90 {
        if i % 3 == 0 {
            drawing_area.fill(&WHITE).unwrap();
            chart
                .configure_mesh()
                .x_desc("x")
                .y_desc("y")
                .draw()
                .unwrap();
            chart
                .draw_series(LineSeries::new(
                    x.iter().zip(v1.iter()).map(|(&x, &y)| (x, y)),
                    &BLACK,
                ))
                .unwrap();
            drawing_area.present().unwrap();
        }
        integrate::<Rk4, _, _, _>(rhs, &v1, &mut v2, &mut 0.0, dt, &mut k);
        std::mem::swap(&mut v1, &mut v2);
    }
}

fn neumann_dirichlet(v: ArrayView1<Float>, v0: Float, vn: Float) {
    let drawing_area = BitMapBackend::gif("neumann_dirichlet.gif", (300, 300), 100)
        .unwrap()
        .into_drawing_area();
    let mut chart = ChartBuilder::on(&drawing_area)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..1.01, -1.0..2.0)
        .unwrap();

    let nx = v.len();
    let dt = 0.2 / nx.pow(2) as Float / 3.0;
    let x = Array1::from_shape_fn((nx,), |i| i as Float / (nx - 1) as Float);

    let op = SBP4;

    let mut k = [v.to_owned(), v.to_owned(), v.to_owned(), v.to_owned()];
    let rhs = move |fut: &mut Array1<Float>, prev: &Array1<Float>, _t: Float| {
        fut.fill(0.0);
        op.diff2(prev.view(), fut.view_mut());

        let h = 1.0 / (nx - 1) as Float;
        let tau = (1.0, -1.0);

        let d1 = op.d1();

        fut[0] += tau.0 / (h * h)
            * (d1
                .iter()
                .zip(prev.iter())
                .map(|(x, y)| x * y)
                .sum::<Float>()
                - v0);
        for (d, fut) in d1.iter().zip(fut.iter_mut().rev().take(d1.len()).rev()) {
            *fut += tau.1 / (h * h) * d * (prev[nx - 1] - vn);
        }
    };

    let mut v1 = v.to_owned();
    let mut v2 = v.to_owned();
    for i in 0..90 {
        if i % 3 == 0 {
            drawing_area.fill(&WHITE).unwrap();
            chart
                .configure_mesh()
                .x_desc("x")
                .y_desc("y")
                .draw()
                .unwrap();
            chart
                .draw_series(LineSeries::new(
                    x.iter().zip(v1.iter()).map(|(&x, &y)| (x, y)),
                    &BLACK,
                ))
                .unwrap();
            drawing_area.present().unwrap();
        }
        integrate::<Rk4, _, _, _>(rhs, &v1, &mut v2, &mut 0.0, dt, &mut k);
        std::mem::swap(&mut v1, &mut v2);
    }
}
