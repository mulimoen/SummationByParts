use ndarray::Array1;
use plotters::prelude::*;
use sbp::{
    integrate::{integrate, Rk4},
    operators::{SbpOperator1d2, SBP4},
    Float,
};

fn main() {
    let drawing_area = BitMapBackend::gif("result.gif", (300, 300), 100)
        .unwrap()
        .into_drawing_area();
    let mut chart = ChartBuilder::on(&drawing_area)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..1.0, -1.0..2.0)
        .unwrap();

    let nx: usize = 101;
    let dt = 0.2 / nx.pow(2) as Float / 3.0;
    let x = Array1::from_shape_fn((nx,), |i| i as Float / (nx - 1) as Float);
    let v0 = x.map(|&x| 0.0 * x * x * x * x);

    let op = SBP4;

    let mut k = [v0.clone(), v0.clone(), v0.clone(), v0.clone()];
    let rhs = move |fut: &mut Array1<Float>, prev: &Array1<Float>, _t: Float| {
        fut.fill(0.0);
        op.diff2(prev.view(), fut.view_mut());

        let h = 1.0 / (nx - 1) as Float;
        let tau = (1.0, -1.0);
        let v0 = 1.0;
        let vn = -1.0;

        let d1 = op.d1();

        for (d, fut) in d1.iter().zip(fut.iter_mut()) {
            *fut += tau.0 / (h * h) * d * (prev[0] - v0);
        }
        for (d, fut) in d1.iter().zip(fut.iter_mut().rev().take(d1.len()).rev()) {
            *fut += tau.1 / (h * h) * d * (prev[nx - 1] - vn);
        }
    };

    let mut v1 = v0.clone();
    let mut v2 = v0;
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
