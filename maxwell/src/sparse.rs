use super::Float;
use sbp::operators::{SbpOperator2d, UpwindOperator2d};
use sbp::utils::kronecker_product;

fn eye(n: usize) -> sprs::CsMat<Float> {
    sprs::CsMat::eye(n)
}

/// Implicit version of the system
/// C u_{t+1} = u_t
pub fn implicit_matrix(rhs: sprs::CsMatView<Float>, dt: Float) -> sprs::CsMat<Float> {
    let n = rhs.rows();
    let f = rhs.map(|x| x * dt);
    &eye(n) - &f
}

fn diagonal(values: &[Float]) -> sprs::CsMat<Float> {
    let values = values.to_vec();
    let indptr = (0..values.len() + 1).collect();
    let indices = (0..values.len()).collect();

    sprs::CsMat::new((values.len(), values.len()), indptr, indices, values)
}

pub struct Implicit {
    pub(crate) rhs: sprs::CsMat<Float>,
    /// Diagonal matrix
    pub(crate) lhs: sprs::CsMat<Float>,
}

/// Assumes self boundaries
pub fn rhs_matrix(op: &dyn SbpOperator2d, grid: &super::Grid) -> Implicit {
    let metrics = grid.metrics(op).unwrap();
    let nx = grid.nx();
    let ny = grid.ny();
    let fluxes = {
        let d1_xi = op.op_eta().diff_matrix(nx);
        let d1_eta = op.op_xi().diff_matrix(ny);

        let d1_xi = kronecker_product(eye(ny).view(), d1_xi.view());
        let d1_eta = kronecker_product(d1_eta.view(), eye(nx).view());

        let mut a_flux = sprs::CsMat::zero((3, 3));
        a_flux.insert(1, 2, -1.0);
        a_flux.insert(2, 1, -1.0);

        let mut b_flux = sprs::CsMat::zero((3, 3));
        b_flux.insert(0, 1, 1.0);
        b_flux.insert(1, 0, 1.0);

        let detj_dxi_dx = diagonal(metrics.detj_dxi_dx().as_slice().unwrap());
        let detj_dxi_dx = &d1_xi * &detj_dxi_dx;
        // Can multiply with the constant matrix after differentiation
        let f_flux_dxi = kronecker_product(a_flux.view(), detj_dxi_dx.view());

        let detj_dxi_dy = diagonal(metrics.detj_dxi_dy().as_slice().unwrap());
        let detj_dxi_dy = &d1_xi * &detj_dxi_dy;
        let f_flux_deta = kronecker_product(b_flux.view(), detj_dxi_dy.view());

        let detj_deta_dx = diagonal(metrics.detj_deta_dx().as_slice().unwrap());
        let detj_deta_dx = &d1_eta * &detj_deta_dx;
        let g_flux_dxi = kronecker_product(a_flux.view(), detj_deta_dx.view());

        let detj_deta_dy = diagonal(metrics.detj_deta_dy().as_slice().unwrap());
        let detj_deta_dy = &d1_eta * &detj_deta_dy;
        let g_flux_deta = kronecker_product(b_flux.view(), detj_deta_dy.view());

        let f_flux = &f_flux_dxi + &f_flux_deta;
        let g_flux = &g_flux_dxi + &g_flux_deta;

        &f_flux + &g_flux
    };

    fn flux_matrix(
        kx: ndarray::ArrayView2<Float>,
        ky: ndarray::ArrayView2<Float>,
        positive: bool,
    ) -> sprs::CsMat<Float> {
        let mut r = &(&kx * &kx) + &(&ky * &ky);
        r.map_inplace(|v| *v = v.sqrt());
        let a00 = if positive {
            &ky * &ky / (2.0 * &r)
        } else {
            -&ky * ky / (2.0 * &r)
        };
        let a00 = diagonal(a00.as_slice().unwrap());
        let a01 = &ky / 2.0;
        let a01 = diagonal(a01.as_slice().unwrap());
        let a02 = &kx * &ky / (2.0 * &r);
        let a02 = diagonal(a02.as_slice().unwrap());
        let a10 = &a01;
        let a11 = if positive { &r / 2.0 } else { -&r / 2.0 };
        let a11 = diagonal(a11.as_slice().unwrap());
        let a12 = -&kx / 2.0;
        let a12 = diagonal(a12.as_slice().unwrap());
        let a20 = &a02;
        let a21 = &a12;
        let a22 = if positive {
            &kx * &kx / (2.0 * &r)
        } else {
            -&kx * kx / (2.0 * &r)
        };
        let a22 = diagonal(a22.as_slice().unwrap());

        sprs::bmat(&[
            [Some(a00.view()), Some(a01.view()), Some(a02.view())],
            [Some(a10.view()), Some(a11.view()), Some(a12.view())],
            [Some(a20.view()), Some(a21.view()), Some(a22.view())],
        ])
    }

    let e0 = |n| {
        let mut e0 = sprs::CsMat::zero((n, 1));
        e0.insert(0, 0, 1.0);
        e0
    };
    let en = |n| {
        let mut en = sprs::CsMat::zero((n, 1));
        en.insert(n - 1, 0, 1.0);
        en
    };

    let sat_west = {
        // West boundary
        let e0 = e0(nx);
        let en = en(nx);

        // Periodic => (e_0 - e_n)q => 0
        let p = &e0 - &en;

        // Forming the matrix of size (nx,nx)
        let mat = &e0 * &p.transpose_view();
        // Must be scaled by the h norm
        let hi = op.op_xi().h_matrix(nx).map(|h| 1.0 / h);
        let mat = &hi * &mat;
        // Upscaling to (nx * ny, nx * ny)
        let mat = kronecker_product(eye(ny).view(), mat.view());

        let aminus = flux_matrix(metrics.detj_dxi_dx(), metrics.detj_dxi_dy(), false);
        let mut sat = &aminus * &kronecker_product(eye(3).view(), mat.view());

        let tau = 1.0;
        // Scaling by tau
        sat.map_inplace(|x| tau * x);
        sat
    };

    let sat_east = {
        // East boundary
        let e0 = e0(nx);
        let en = en(nx);

        // Periodic => (e_0 - e_n) => 0
        let p = &en - &e0;

        // Forming the matrix of size (nx,nx)
        let mat = &en * &p.transpose_view();
        // Must be scaled by the h norm
        let hi = op.op_xi().h_matrix(nx).map(|h| 1.0 / h);
        let mat = &hi * &mat;
        // Upscaling to (nx * ny, nx * ny)
        let mat = kronecker_product(eye(ny).view(), mat.view());

        let aplus = flux_matrix(metrics.detj_dxi_dx(), metrics.detj_dxi_dy(), true);

        let mut sat = &aplus * &kronecker_product(eye(3).view(), mat.view());

        let tau = -1.0;
        // Scaling by tau
        sat.map_inplace(|x| tau * x);
        sat
    };

    let sat_south = {
        // South boundary
        let e0 = e0(ny);
        let en = en(ny);

        // Periodic => (e_0 - e_n) => 0
        let p = &e0 - &en;

        // Forming the matrix of size (ny,ny)
        let mat = &e0 * &p.transpose_view();
        // Must be scaled by the h norm
        let hi = op.op_eta().h_matrix(ny).map(|h| 1.0 / h);
        let mat = &hi * &mat;
        // Upscaling to (nx * ny, nx * ny)
        let mat = kronecker_product(mat.view(), eye(nx).view());

        let bminus = flux_matrix(metrics.detj_deta_dx(), metrics.detj_deta_dy(), false);

        let mut sat = &bminus * &kronecker_product(eye(3).view(), mat.view());

        let tau = 1.0;
        // Scaling by tau
        sat.map_inplace(|x| tau * x);
        sat
    };

    let sat_north = {
        // North boundary
        let e0 = e0(ny);
        let en = en(ny);

        // Periodic => (e_0 - e_n) => 0
        let p = &en - &e0;

        // Forming the matrix of size (ny,ny)
        let mat = &en * &p.transpose_view();
        // Must be scaled by the h norm
        let hi = op.op_eta().h_matrix(ny).map(|h| 1.0 / h);
        let mat = &hi * &mat;
        // Upscaling to (nx * ny, nx * ny)
        let mat = kronecker_product(mat.view(), eye(nx).view());

        let bminus = flux_matrix(metrics.detj_deta_dx(), metrics.detj_deta_dy(), true);

        let mut sat = &bminus * &kronecker_product(eye(3).view(), mat.view());

        let tau = -1.0;
        // Scaling by tau
        sat.map_inplace(|x| tau * x);
        sat
    };

    let rhs = &fluxes + &(&(&sat_west + &sat_east) + &(&sat_north + &sat_south));

    Implicit {
        rhs,
        lhs: kronecker_product(
            sprs::CsMat::eye(3).view(),
            diagonal(metrics.detj().as_slice().unwrap()).view(),
        ),
    }
}

/// RHS with some additional dissipation from the upwind operator
pub fn rhs_matrix_with_upwind_dissipation(
    op: impl UpwindOperator2d + SbpOperator2d,
    grid: &super::Grid,
) -> sprs::CsMat<Float> {
    let rhs = rhs_matrix(&op, grid).rhs;
    let metrics = grid.metrics(&op).unwrap();
    let nx = grid.nx();
    let ny = grid.ny();

    let diss = |kx: ndarray::ArrayView2<Float>, ky: ndarray::ArrayView2<Float>| {
        let r = &kx * &kx + &ky * &ky;
        let s00 = &ky * &ky / &r;
        let s00 = diagonal(s00.as_slice().unwrap());
        let s02 = -&kx * ky / &r;
        let s02 = diagonal(s02.as_slice().unwrap());
        let s11 = diagonal(r.as_slice().unwrap());
        let s20 = &s02;
        let s22 = &kx * &kx / &r;
        let s22 = diagonal(s22.as_slice().unwrap());
        sprs::bmat(&[
            [Some(s00.view()), None, Some(s02.view())],
            [None, Some(s11.view()), None],
            [Some(s20.view()), None, Some(s22.view())],
        ])
    };

    let diss_x = {
        let diss_x = UpwindOperator2d::op_xi(&op).diss_matrix(nx);
        let diss_x = kronecker_product(eye(ny).view(), diss_x.view());
        let met = diss(metrics.detj_dxi_dx(), metrics.detj_dxi_dy());
        &met * &kronecker_product(eye(3).view(), diss_x.view())
    };

    let diss_y = {
        let diss_y = UpwindOperator2d::op_eta(&op).diss_matrix(ny);
        let diss_y = kronecker_product(diss_y.view(), eye(nx).view());
        let met = diss(metrics.detj_deta_dx(), metrics.detj_deta_dy());
        &met * &kronecker_product(eye(3).view(), diss_y.view())
    };

    &rhs + &(&diss_x + &diss_y)
}

#[test]
fn creation() {
    let ny = 16;
    let nx = 170;

    let x = ndarray::Array::from_shape_fn((ny, nx), |(_j, i)| i as Float / (nx - 1) as Float);
    let y = ndarray::Array::from_shape_fn((ny, nx), |(j, _i)| j as Float / (ny - 1) as Float);

    let op = &sbp::operators::Upwind4;

    let grid = sbp::grid::Grid::new(x, y).unwrap();

    let _rhs = rhs_matrix(op, &grid);
    // let _lhs = implicit_matrix(rhs.view(), 1e-2);
    let _rhs_upwind = rhs_matrix_with_upwind_dissipation(op, &grid);
}
