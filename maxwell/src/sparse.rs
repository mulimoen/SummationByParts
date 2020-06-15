use super::Float;
use sbp::operators::SbpOperator2d;
use sbp::utils::sparse_sparse_outer_product;

/// Implicit version of the system
/// C u_{t+1} = u_t
pub fn implicit_matrix(rhs: sprs::CsMatView<Float>, dt: Float) -> sprs::CsMat<Float> {
    let n = rhs.rows();
    let i_kyx = sprs::CsMat::eye(n);
    let f = rhs.map(|x| x * dt);
    &i_kyx - &f
}

/// Assumes self boundaries
pub fn rhs_matrix(op: &dyn SbpOperator2d, ny: usize, nx: usize) -> sprs::CsMat<Float> {
    let eye = |n: usize| sprs::CsMat::eye(n);

    let fluxes = {
        let d1_x = op.op_eta().diff_matrix(nx);
        let d1_y = op.op_xi().diff_matrix(ny);

        let dx = sparse_sparse_outer_product(eye(ny).view(), d1_x.view());
        let dy = sparse_sparse_outer_product(d1_y.view(), eye(nx).view());

        let mut a_flux = sprs::CsMat::zero((3, 3));
        a_flux.insert(1, 2, -1.0);
        a_flux.insert(2, 1, -1.0);

        let mut b_flux = sprs::CsMat::zero((3, 3));
        b_flux.insert(0, 1, 1.0);
        b_flux.insert(1, 0, 1.0);

        &sparse_sparse_outer_product(a_flux.view(), dx.view())
            + &sparse_sparse_outer_product(b_flux.view(), dy.view())
    };

    let sat_west = {
        // West boundary
        let aminus = {
            let mut aminus = sprs::CsMat::zero((3, 3));
            aminus.insert(1, 1, -0.5);
            aminus.insert(1, 2, -0.5);
            aminus.insert(2, 1, -0.5);
            aminus.insert(2, 2, -0.5);
            aminus
        };
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

        // Periodic => (e_0 - e_n)q => 0
        let p = &e0 - &en;

        // Forming the matrix of size (nx,nx)
        let mat = &e0 * &p.transpose_view();
        // Must be scaled by the h norm
        let hi = op.op_xi().h_matrix(nx).map(|h| 1.0 / h);
        let mat = &hi * &mat;
        // Upscaling to (nx * ny, nx * ny)
        let mat = sparse_sparse_outer_product(eye(ny).view(), mat.view());

        let mut sat = sparse_sparse_outer_product(aminus.view(), mat.view());

        let tau = 1.0;
        // Scaling by tau
        sat.map_inplace(|x| tau * x);
        sat
    };

    let sat_east = {
        // East boundary
        let aminus = {
            let mut aplus = sprs::CsMat::zero((3, 3));
            aplus.insert(1, 1, 0.5);
            aplus.insert(1, 2, -0.5);
            aplus.insert(2, 1, -0.5);
            aplus.insert(2, 2, 0.5);
            aplus
        };
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

        // Periodic => (e_0 - e_n) => 0
        let p = &en - &e0;

        // Forming the matrix of size (nx,nx)
        let mat = &en * &p.transpose_view();
        // Must be scaled by the h norm
        let hi = op.op_xi().h_matrix(nx).map(|h| 1.0 / h);
        let mat = &hi * &mat;
        // Upscaling to (nx * ny, nx * ny)
        let mat = sparse_sparse_outer_product(eye(ny).view(), mat.view());

        let mut sat = sparse_sparse_outer_product(aminus.view(), mat.view());

        let tau = -1.0;
        // Scaling by tau
        sat.map_inplace(|x| tau * x);
        sat
    };

    let sat_south = {
        // South boundary
        let bminus = {
            let mut bminus = sprs::CsMat::zero((3, 3));
            bminus.insert(0, 0, -0.5);
            bminus.insert(0, 1, 0.5);
            bminus.insert(1, 0, 0.5);
            bminus.insert(1, 1, -0.5);
            bminus
        };
        let e0 = {
            let mut e0 = sprs::CsMat::zero((ny, 1));
            e0.insert(0, 0, 1.0);
            e0
        };
        let en = {
            let mut en = sprs::CsMat::zero((ny, 1));
            en.insert(ny - 1, 0, 1.0);
            en
        };

        // Periodic => (e_0 - e_n) => 0
        let p = &e0 - &en;

        // Forming the matrix of size (ny,ny)
        let mat = &e0 * &p.transpose_view();
        // Must be scaled by the h norm
        let hi = op.op_eta().h_matrix(ny).map(|h| 1.0 / h);
        let mat = &hi * &mat;
        // Upscaling to (nx * ny, nx * ny)
        let mat = sparse_sparse_outer_product(mat.view(), eye(nx).view());

        let mut sat = sparse_sparse_outer_product(bminus.view(), mat.view());

        let tau = 1.0;
        // Scaling by tau
        sat.map_inplace(|x| tau * x);
        sat
    };

    let sat_north = {
        // North boundary
        let bplus = {
            let mut bplus = sprs::CsMat::zero((3, 3));
            bplus.insert(0, 0, 0.5);
            bplus.insert(0, 1, 0.5);
            bplus.insert(1, 0, 0.5);
            bplus.insert(1, 1, 0.5);
            bplus
        };
        let e0 = {
            let mut e0 = sprs::CsMat::zero((ny, 1));
            e0.insert(0, 0, 1.0);
            e0
        };
        let en = {
            let mut en = sprs::CsMat::zero((ny, 1));
            en.insert(ny - 1, 0, 1.0);
            en
        };

        // Periodic => (e_0 - e_n) => 0
        let p = &en - &e0;

        // Forming the matrix of size (ny,ny)
        let mat = &en * &p.transpose_view();
        // Must be scaled by the h norm
        let hi = op.op_eta().h_matrix(ny).map(|h| 1.0 / h);
        let mat = &hi * &mat;
        // Upscaling to (nx * ny, nx * ny)
        let mat = sparse_sparse_outer_product(mat.view(), eye(nx).view());

        let mut sat = sparse_sparse_outer_product(bplus.view(), mat.view());

        let tau = -1.0;
        // Scaling by tau
        sat.map_inplace(|x| tau * x);
        sat
    };

    &fluxes + &(&(&sat_west + &sat_east) + &(&sat_north + &sat_south))
}

#[test]
fn dummy() {
    let ny = 16;
    let nx = 17;

    let rhs = rhs_matrix(&sbp::operators::Upwind4, ny, nx);
    let _lhs = implicit_matrix(rhs.view(), 1e-2);
}
