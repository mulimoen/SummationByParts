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
    let d1_x = op.op_eta().diff_matrix(nx);
    let d1_y = op.op_xi().diff_matrix(ny);

    let ix = sprs::CsMat::<Float>::eye(nx);
    let iy = sprs::CsMat::eye(ny);

    let dx = sparse_sparse_outer_product(iy.view(), d1_x.view());
    let dy = sparse_sparse_outer_product(d1_y.view(), ix.view());

    let mut a_flux = sprs::TriMat::new((3, 3));
    a_flux.add_triplet(1, 2, -1.0);
    a_flux.add_triplet(2, 1, -1.0);
    let a_flux = a_flux.to_csr();

    let mut b_flux = sprs::TriMat::new((3, 3));
    b_flux.add_triplet(0, 1, 1.0);
    b_flux.add_triplet(1, 0, 1.0);
    let b_flux = b_flux.to_csr();

    let f = &sparse_sparse_outer_product(a_flux.view(), dx.view())
        + &sparse_sparse_outer_product(b_flux.view(), dy.view());

    let mut hx = sparse_sparse_outer_product(iy.view(), op.op_xi().h_matrix(nx).view());
    hx.map_inplace(|h| 1.0 / h);
    let ihx = hx;
    let mut hy = sparse_sparse_outer_product(op.op_eta().h_matrix(ny).view(), ix.view());
    hy.map_inplace(|h| 1.0 / h);
    let ihy = hy;

    let _f = {
        // West boundary
        let mut aminus = sprs::TriMat::new((3, 3));
        aminus.add_triplet(1, 1, -0.5);
        aminus.add_triplet(1, 2, -0.5);
        aminus.add_triplet(2, 1, -0.5);
        aminus.add_triplet(2, 2, -0.5);
        let aminus = aminus.to_csr();

        let mut e0x = sprs::TriMat::new((nx, 1));
        e0x.add_triplet(0, 0, 1.0);
        let e0x = e0x.to_csr();
        let e0x_nt = &e0x * &e0x.transpose_view();
        let e0x_nt = sparse_sparse_outer_product(iy.view(), e0x_nt.view());

        let sat0 = &ihx * &e0x_nt;
        let mut sat0 = sparse_sparse_outer_product(aminus.view(), sat0.view());

        let tau = 1.0;
        sat0.map_inplace(|x| tau * x);

        &f + &sat0
    };

    let _f = {
        // East boundary
        let mut aplus = sprs::TriMat::new((3, 3));
        aplus.add_triplet(1, 1, 0.5);
        aplus.add_triplet(1, 2, -0.5);
        aplus.add_triplet(2, 1, -0.5);
        aplus.add_triplet(2, 2, 0.5);
        let aplus = aplus.to_csr();

        let mut enx = sprs::TriMat::new((nx, 1));
        enx.add_triplet(nx - 1, 0, 1.0);
        let enx = enx.to_csr();
        let enx_nt = &enx * &enx.transpose_view();
        let enx_nt = sparse_sparse_outer_product(iy.view(), enx_nt.view());

        let satn = &ihx * &enx_nt;
        let mut satn = sparse_sparse_outer_product(aplus.view(), satn.view());

        let tau = -1.0;
        satn.map_inplace(|x| tau * x);

        &f + &satn
    };

    let _f = {
        // South boundary
        let mut bminus = sprs::TriMat::new((3, 3));
        bminus.add_triplet(0, 0, -0.5);
        bminus.add_triplet(0, 1, 0.5);
        bminus.add_triplet(1, 0, 0.5);
        bminus.add_triplet(1, 1, -0.5);
        let bminus = bminus.to_csr();

        let mut e0y = sprs::TriMat::new((ny, 1));
        e0y.add_triplet(0, 0, 1.0);
        let e0y = e0y.to_csr();
        let e0y_nt = &e0y * &e0y.transpose_view();
        let e0y_nt = sparse_sparse_outer_product(e0y_nt.view(), ix.view());

        let sat0 = &ihx * &e0y_nt;
        let mut sat0 = sparse_sparse_outer_product(bminus.view(), sat0.view());

        let tau = 1.0;
        sat0.map_inplace(|x| tau * x);

        &f + &sat0
    };

    let _f = {
        // North boundary
        let mut bplus = sprs::TriMat::new((3, 3));
        bplus.add_triplet(0, 0, 0.5);
        bplus.add_triplet(0, 1, 0.5);
        bplus.add_triplet(1, 0, 0.5);
        bplus.add_triplet(1, 1, 0.5);
        let bplus = bplus.to_csr();

        let mut eny = sprs::TriMat::new((ny, 1));
        eny.add_triplet(ny - 1, 0, 1.0);
        let eny = eny.to_csr();
        let eny_nt = &eny * &eny.transpose_view();
        let eny_nt = sparse_sparse_outer_product(eny_nt.view(), ix.view());

        let satn = &ihy * &eny_nt;
        let mut satn = sparse_sparse_outer_product(bplus.view(), satn.view());

        let tau = -1.0;
        satn.map_inplace(|x| tau * x);

        &f + &satn
    };
    // Setting up the periodic boundaries
    let _f = {
        // West
        let mut aminus = sprs::TriMat::new((3, 3));
        aminus.add_triplet(1, 1, -0.5);
        aminus.add_triplet(1, 2, -0.5);
        aminus.add_triplet(2, 1, -0.5);
        aminus.add_triplet(2, 2, -0.5);
        let aminus = aminus.to_csr();

        let mut e0x = sprs::TriMat::new((nx, 1));
        e0x.add_triplet(0, 0, 1.0);
        let e0x = e0x.to_csr();
        let mut enx = sprs::TriMat::new((nx, 1));
        enx.add_triplet(nx - 1, 0, 1.0);
        let enx = enx.to_csr();

        let e0nx_nt = &e0x * &enx.transpose_view();
        let e0nx_nt = sparse_sparse_outer_product(iy.view(), e0nx_nt.view());

        let sat0 = &ihx * &e0nx_nt;
        let mut sat0 = sparse_sparse_outer_product(aminus.view(), sat0.view());

        let tau = 1.0;
        // Negative => subtracting this boundary
        sat0.map_inplace(|x| -tau * x);

        &f + &sat0
    };

    let _f = {
        // East boundary
        let mut aplus = sprs::TriMat::new((3, 3));
        aplus.add_triplet(1, 1, 0.5);
        aplus.add_triplet(1, 2, -0.5);
        aplus.add_triplet(2, 1, -0.5);
        aplus.add_triplet(2, 2, 0.5);
        let aplus = aplus.to_csr();

        let mut enx = sprs::TriMat::new((nx, 1));
        enx.add_triplet(nx - 1, 0, 1.0);
        let enx = enx.to_csr();
        let mut e0x = sprs::TriMat::new((nx, 1));
        e0x.add_triplet(0, 0, 1.0);
        let e0x = e0x.to_csr();

        let en0x_nt = &enx * &e0x.transpose_view();
        let en0x_nt = sparse_sparse_outer_product(iy.view(), en0x_nt.view());

        let satn = &ihx * &en0x_nt;
        let mut satn = sparse_sparse_outer_product(aplus.view(), satn.view());

        let tau = -1.0;
        satn.map_inplace(|x| -tau * x);

        &f + &satn
    };

    let _f = {
        // South boundary
        let mut bminus = sprs::TriMat::new((3, 3));
        bminus.add_triplet(0, 0, -0.5);
        bminus.add_triplet(0, 1, 0.5);
        bminus.add_triplet(1, 0, 0.5);
        bminus.add_triplet(1, 1, -0.5);
        let bminus = bminus.to_csr();

        let mut e0y = sprs::TriMat::new((ny, 1));
        e0y.add_triplet(0, 0, 1.0);
        let e0y = e0y.to_csr();
        let mut eny = sprs::TriMat::new((ny, 1));
        eny.add_triplet(ny - 1, 0, 1.0);
        let eny = eny.to_csr();

        let e0ny_nt = &e0y * &eny.transpose_view();
        let e0ny_nt = sparse_sparse_outer_product(e0ny_nt.view(), ix.view());

        let sat0 = &ihx * &e0ny_nt;
        let mut sat0 = sparse_sparse_outer_product(bminus.view(), sat0.view());

        let tau = 1.0;
        sat0.map_inplace(|x| -tau * x);

        &f + &sat0
    };

    let _f = {
        // North boundary
        let mut bplus = sprs::TriMat::new((3, 3));
        bplus.add_triplet(0, 0, 0.5);
        bplus.add_triplet(0, 1, 0.5);
        bplus.add_triplet(1, 0, 0.5);
        bplus.add_triplet(1, 1, 0.5);
        let bplus = bplus.to_csr();

        let mut eny = sprs::TriMat::new((ny, 1));
        eny.add_triplet(ny - 1, 0, 1.0);
        let eny = eny.to_csr();
        let mut e0y = sprs::TriMat::new((ny, 1));
        e0y.add_triplet(0, 0, 1.0);
        let e0y = e0y.to_csr();

        let en0y_nt = &eny * &e0y.transpose_view();
        let en0y_nt = sparse_sparse_outer_product(en0y_nt.view(), ix.view());

        let satn = &ihy * &en0y_nt;
        let mut satn = sparse_sparse_outer_product(bplus.view(), satn.view());

        let tau = -1.0;
        satn.map_inplace(|x| -tau * x);

        &f + &satn
    };
    f
}
