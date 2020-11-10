use ndarray::{Array1, ArrayView1};
use plotters::prelude::*;
use sbp::{
    integrate::{integrate, Rk4},
    operators::{SbpOperator1d, SbpOperator1d2, SBP4},
    Float,
};

fn main() {
    let nx: usize = 101;
    let x = Array1::from_shape_fn((nx,), |i| i as Float / (nx - 1) as Float);
    let v0 = x.map(|&x| (-(x - 0.5).powi(2) / 0.1).exp());
    dual_dirichlet(v0.view(), 1.0, 1.0);
    // Neumann boundary is introducing energy into the system
    neumann_dirichlet(v0.view(), -0.2, 1.0);

    // The sparse formulation
    dual_dirichlet_sparse(v0.view(), 1.0, 1.0);
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
        for (d, fut) in d1
            .iter()
            .rev()
            .map(|d| -d)
            .zip(fut.iter_mut().rev().take(d1.len()).rev())
        {
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
        for (d, fut) in d1
            .iter()
            .rev()
            .map(|d| -d)
            .zip(fut.iter_mut().rev().take(d1.len()).rev())
        {
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

fn dual_dirichlet_sparse(v: ArrayView1<Float>, v0: Float, vn: Float) {
    let drawing_area = BitMapBackend::gif("dual_dirichlet_sparse.gif", (300, 300), 100)
        .unwrap()
        .into_drawing_area();
    let mut chart = ChartBuilder::on(&drawing_area)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..1.01, -1.0..2.0)
        .unwrap();

    let op = SBP4;
    let nx = v.len();
    let system = op.diff2_matrix(nx);
    let e0 = {
        let mut e0 = sprs::CsMat::zero((nx, 1));
        e0.insert(0, 0, 1.0);
        e0
    };
    let en = {
        let mut en = sprs::CsMat::zero((nx, 1));
        en.insert(nx - 1, 0, 1.0);
        en
    };

    let mut hi = op.h_matrix(nx);
    hi.map_inplace(|v| 1.0 / v);

    let tau = (1.0, -1.0);
    let sat0 = {
        let d1 = op.d1_vec(nx, true);
        let mut mat = &hi * &d1.transpose_view();
        mat.map_inplace(|v| v * tau.0);
        (&mat * &e0.transpose_view(), mat)
    };
    let satn = {
        let dn = op.d1_vec(nx, false);
        let mut mat = &hi * &dn.transpose_view();
        mat.map_inplace(|v| v * tau.1);
        (&mat * &en.transpose_view(), mat)
    };

    let system = &system + &(&sat0.0 + &satn.0);
    // Stack the two matrices to allow easy book-keeping
    // of boundary conditions
    let mut bc = sprs::hstack(&[sat0.1.view(), satn.1.view()]).to_csr();
    bc.map_inplace(|v| -v);
    let system = &system;
    let bc = &bc;

    let dt = 0.2 / nx.pow(2) as Float / 3.0;
    let x = Array1::from_shape_fn((nx,), |i| i as Float / (nx - 1) as Float);

    let mut k = [v.to_owned(), v.to_owned(), v.to_owned(), v.to_owned()];
    let rhs = move |fut: &mut Array1<Float>, prev: &Array1<Float>, _t: Float| {
        fut.fill(0.0);
        let prev = prev.as_slice().unwrap();
        let fut = fut.as_slice_mut().unwrap();

        sprs::prod::mul_acc_mat_vec_csr(system.view(), prev, fut);

        sprs::prod::mul_acc_mat_vec_csr(bc.view(), &[v0, vn][..], fut);
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

#[test]
fn eigenvalues_diri_neumann() {
    let op = SBP4;
    let nx = 50;

    let v0 = 0.0;
    let vn = 0.0;

    // Test eigenvalue
    let lhs = |p: &[f64], f: &mut [f64]| {
        let mut fut = ndarray::ArrayViewMut::from_shape(nx, f).unwrap();
        let prev = ndarray::ArrayView::from_shape(nx, p).unwrap();

        fut.fill(0.0);
        op.diff2(prev.view(), fut.view_mut());

        let h = 1.0 / (nx - 1) as Float;
        let tau = (1.0, -1.0);

        let d1 = op.d1();

        for (d, fut) in d1.iter().zip(fut.iter_mut()) {
            *fut += tau.0 / (h * h) * d * (prev[0] - v0);
        }
        for (d, fut) in d1
            .iter()
            .rev()
            .map(|d| -d)
            .zip(fut.iter_mut().rev().take(d1.len()).rev())
        {
            *fut += tau.1 / (h * h) * d * (prev[nx - 1] - vn);
        }
    };
    let (lambda, _) = arpack::dnaupd(
        lhs,
        arpack::InputParameters {
            which: arpack::Which::LargestRealpart,
            n: nx,
            nev: 3,
            ncv: 40,
            mxiter: 500,
            ..Default::default()
        },
    );
    assert!(lambda.0.iter().all(|&x| x <= 0.0));
}
