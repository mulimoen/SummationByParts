use maxwell::Field;
use ndarray::Array2;
use sbp::{operators::SbpOperator1d, Float};

struct SparseMaxwellSystem {
    x: Array2<Float>,
    y: Array2<Float>,
    rhs: sprs::CsMat<Float>,
    lhs_implicit: Option<sprs::CsMat<Float>>,
    now: Field,
    next: Field,
    k: [Field; 4],
}

impl SparseMaxwellSystem {
    fn new(ny: usize, nx: usize) -> Self {
        let x = ndarray::Array::from_shape_fn((ny, nx), |(_j, i)| {
            i as Float * (1.0 / (nx - 1) as Float)
        });
        let y = ndarray::Array::from_shape_fn((ny, nx), |(j, _i)| {
            j as Float * (1.0 / (nx - 1) as Float)
        });

        let rhs = Self::make_matrix(ny, nx);

        let mut now = Field::new(ny, nx);
        let mut next = Field::new(ny, nx);
        let mut k = [now.clone(), now.clone(), now.clone(), now.clone()];

        Self {
            x,
            y,
            rhs,
            now,
            next,
            k,
            lhs_implicit: None,
        }
    }
    fn nx(&self) -> usize {
        self.x.shape()[1]
    }
    fn ny(&self) -> usize {
        self.x.shape()[0]
    }
    fn max_dt(&self) -> Float {
        1.0 / std::cmp::max(self.nx(), self.ny()) as Float
    }
    fn make_matrix(ny: usize, nx: usize) -> sprs::CsMat<Float> {
        let d1_x = sbp::operators::Upwind4.diff_matrix(nx);
        let d1_y = sbp::operators::Upwind4.diff_matrix(ny);

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

        let mut hx =
            sparse_sparse_outer_product(iy.view(), sbp::operators::Upwind4.h_matrix(nx).view());
        hx.map_inplace(|h| 1.0 / h);
        let ihx = hx;
        let mut hy =
            sparse_sparse_outer_product(sbp::operators::Upwind4.h_matrix(ny).view(), ix.view());
        hy.map_inplace(|h| 1.0 / h);
        let ihy = hy;

        let f = {
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

        let f = {
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

        let f = {
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

        let f = {
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
        let f = {
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

        let f = {
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

        let f = {
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

        let f = {
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
    fn advance(&mut self) {
        let max_dt = self.max_dt();
        let rhs = self.rhs.view();
        let rhs_f = |next: &mut Field, now: &Field, _t: Float| {
            next.fill(0.0);
            sprs::prod::mul_acc_mat_vec_csr(
                rhs,
                now.as_slice().unwrap(),
                next.as_slice_mut().unwrap(),
            );
        };
        sbp::integrate::integrate::<sbp::integrate::Rk4, _, _>(
            rhs_f,
            &self.now,
            &mut self.next,
            &mut 0.0,
            max_dt,
            &mut self.k[..],
        );
        std::mem::swap(&mut self.now, &mut self.next);
    }
    fn advance_implicit(&mut self) {
        if self.lhs_implicit.is_none() {
            self.lhs_implicit = Some({
                let i_kyx = sprs::CsMat::eye(3 * self.ny() * self.nx());
                let dt = self.max_dt();
                let f = self.rhs.map(|x| x * dt);
                &i_kyx - &f
            });
        }
        let b = self.now.clone();

        let tnow = std::time::Instant::now();
        jacobi_method(
            self.lhs_implicit.as_ref().unwrap().view(),
            b.as_slice().unwrap(),
            self.now.as_slice_mut().unwrap(),
            self.next.as_slice_mut().unwrap(),
            10,
        );
        let elapsed = tnow.elapsed();
        println!("{:?}", elapsed);
    }
}

fn main() {
    let nx = 64;
    let ny = 64;

    let mut sys = SparseMaxwellSystem::new(ny, nx);

    let to_image = |mat: sprs::CsMatView<Float>, path: &str| {
        let sparsity = sprs::visu::nnz_image(mat);
        let im: image::ImageBuffer<image::Luma<u8>, _> = sparsity
            .as_slice()
            .map(|slice| {
                image::ImageBuffer::from_raw((3 * nx * ny) as u32, (3 * nx * ny) as u32, slice)
                    .expect("failed to create image from slice")
            })
            .unwrap();
        im.save(path).unwrap();
    };

    let tnow = std::time::Instant::now();
    for _ in 0..100 {
        sys.advance();
    }
    let elapsed = tnow.elapsed();
    println!("{:?}", elapsed.div_f64(100.0));
}

/// A x = b
/// with A and b known
/// x should contain a first guess of
fn jacobi_method(
    a: sprs::CsMatView<Float>,
    b: &[Float],
    x: &mut [Float],
    tmp: &mut [Float],
    iter_count: usize,
) {
    for _ in 0..iter_count {
        jacobi_step(a, b, x, tmp);
        x.copy_from_slice(tmp);
    }
}

fn jacobi_step(a: sprs::CsMatView<Float>, b: &[Float], x0: &[Float], x: &mut [Float]) {
    let n = a.shape().0;
    assert_eq!(n, a.shape().1);
    let b = &b[..n];
    let x0 = &x0[..n];
    let x = &mut x[..n];
    for (((i, ai), xi), &bi) in a
        .outer_iterator()
        .enumerate()
        .zip(x.iter_mut())
        .zip(b.iter())
    {
        let mut summa = 0.0;
        let mut aii = None;
        for (j, aij) in ai.iter() {
            if i == j {
                aii = Some(aij);
                continue;
            }
            summa += aij * x0[j];
        }
        *xi = 1.0 / aii.unwrap() * (bi - summa);
    }
}

#[test]
fn test_jacobi_2x2() {
    let mut a = sprs::CsMat::zero((2, 2));
    a.insert(0, 0, 2.0);
    a.insert(0, 1, 1.0);
    a.insert(1, 0, 5.0);
    a.insert(1, 1, 7.0);

    let b = ndarray::arr1(&[11.0, 13.0]);

    let mut x0 = ndarray::arr1(&[1.0; 2]);
    let mut tmp = x0.clone();

    jacobi_method(
        a.view(),
        b.as_slice().unwrap(),
        x0.as_slice_mut().unwrap(),
        tmp.as_slice_mut().unwrap(),
        25,
    );

    approx::assert_abs_diff_eq!(x0, ndarray::arr1(&[7.111, -3.222]), epsilon = 1e-2);
}

#[test]
fn test_jacobi_4x4() {
    let mut a = sprs::CsMat::zero((4, 4));
    a.insert(0, 0, 10.0);
    a.insert(0, 1, -1.0);
    a.insert(0, 2, 2.0);
    a.insert(1, 0, -1.0);
    a.insert(1, 1, 11.0);
    a.insert(1, 2, -1.0);
    a.insert(1, 3, 3.0);
    a.insert(2, 0, 2.0);
    a.insert(2, 1, -1.0);
    a.insert(2, 2, 10.0);
    a.insert(2, 3, -1.0);
    a.insert(3, 1, 3.0);
    a.insert(3, 2, -1.0);
    a.insert(3, 3, 8.0);

    let b = ndarray::arr1(&[6.0, 25.0, -11.0, 15.0]);

    let mut x0 = ndarray::Array::zeros(b.len());
    let mut tmp = x0.clone();

    for iter in 0.. {
        jacobi_step(
            a.view(),
            b.as_slice().unwrap(),
            x0.as_slice().unwrap(),
            tmp.as_slice_mut().unwrap(),
        );
        x0.as_slice_mut()
            .unwrap()
            .copy_from_slice(tmp.as_slice().unwrap());
        match iter {
            0 => approx::assert_abs_diff_eq!(
                x0,
                ndarray::arr1(&[0.6, 2.27272, -1.1, 1.875]),
                epsilon = 1e-4
            ),
            1 => approx::assert_abs_diff_eq!(
                x0,
                ndarray::arr1(&[1.04727, 1.7159, -0.80522, 0.88522]),
                epsilon = 1e-4
            ),
            2 => approx::assert_abs_diff_eq!(
                x0,
                ndarray::arr1(&[0.93263, 2.05330, -1.0493, 1.13088]),
                epsilon = 1e-4
            ),
            3 => approx::assert_abs_diff_eq!(
                x0,
                ndarray::arr1(&[1.01519, 1.95369, -0.9681, 0.97384]),
                epsilon = 1e-4
            ),
            4 => approx::assert_abs_diff_eq!(
                x0,
                ndarray::arr1(&[0.98899, 2.0114, -1.0102, 1.02135]),
                epsilon = 1e-4
            ),
            _ => break,
        }
    }
}

/// Computes the sparse kronecker product
/// M = A \kron B
#[allow(non_snake_case)]
#[must_use]
fn sparse_sparse_outer_product<
    N: num_traits::Num + Copy + Default,
    I: sprs::SpIndex,
    Iptr: sprs::SpIndex,
>(
    A: sprs::CsMatViewI<N, I, Iptr>,
    B: sprs::CsMatViewI<N, I, Iptr>,
) -> sprs::CsMatI<N, I, Iptr> {
    match (A.storage(), B.storage()) {
        (sprs::CompressedStorage::CSR, sprs::CompressedStorage::CSR) => {
            let nnz = A.nnz() * B.nnz();
            let a_shape = A.shape();
            let b_shape = B.shape();
            let shape = (a_shape.0 * b_shape.0, a_shape.1 * b_shape.1);
            let mut mat = sprs::CsMatI::zero(shape);
            mat.reserve_nnz_exact(nnz);
            for (aj, a) in A.outer_iterator().enumerate() {
                for (bj, b) in B.outer_iterator().enumerate() {
                    for (ai, &a) in a.iter() {
                        for (bi, &b) in b.iter() {
                            let i = ai * b_shape.1 + bi;
                            let j = aj * b_shape.0 + bj;
                            mat.insert(j, i, a * b)
                        }
                    }
                }
            }
            debug_assert_eq!(mat.nnz(), nnz);
            mat
        }
        (sprs::CompressedStorage::CSC, sprs::CompressedStorage::CSC) => {
            let nnz = A.nnz() * B.nnz();
            let a_shape = A.shape();
            let b_shape = B.shape();
            let shape = (a_shape.0 * b_shape.0, a_shape.1 * b_shape.1);
            let mat = sprs::CsMatI::zero(shape);
            let mut mat = mat.to_csc();

            for (ai, a) in A.outer_iterator().enumerate() {
                for (bi, b) in B.outer_iterator().enumerate() {
                    for (aj, &a) in a.iter() {
                        for (bj, &b) in b.iter() {
                            let i = ai * b_shape.1 + bi;
                            let j = aj * b_shape.0 + bj;
                            mat.insert(j, i, a * b)
                        }
                    }
                }
            }
            debug_assert_eq!(mat.nnz(), nnz);
            mat
        }
        (sprs::CompressedStorage::CSR, sprs::CompressedStorage::CSC) => {
            let nnz = A.nnz() * B.nnz();
            let a_shape = A.shape();
            let b_shape = B.shape();
            let shape = (a_shape.0 * b_shape.0, a_shape.1 * b_shape.1);
            let mut mat = sprs::CsMatI::zero(shape);
            mat.reserve_nnz_exact(nnz);
            for (aj, a) in A.outer_iterator().enumerate() {
                for (bi, b) in B.outer_iterator().enumerate() {
                    for (ai, &a) in a.iter() {
                        for (bj, &b) in b.iter() {
                            let i = ai * b_shape.1 + bi;
                            let j = aj * b_shape.0 + bj;
                            mat.insert(j, i, a * b)
                        }
                    }
                }
            }
            debug_assert_eq!(mat.nnz(), nnz);
            mat
        }
        (sprs::CompressedStorage::CSC, sprs::CompressedStorage::CSR) => {
            let nnz = A.nnz() * B.nnz();
            let a_shape = A.shape();
            let b_shape = B.shape();
            let shape = (a_shape.0 * b_shape.0, a_shape.1 * b_shape.1);
            let mat = sprs::CsMatI::zero(shape);
            let mut mat = mat.to_csc();

            for (aj, a) in A.outer_iterator().enumerate() {
                for (bi, b) in B.outer_iterator().enumerate() {
                    for (ai, &a) in a.iter() {
                        for (bj, &b) in b.iter() {
                            let i = ai * b_shape.1 + bi;
                            let j = aj * b_shape.0 + bj;
                            mat.insert(j, i, a * b)
                        }
                    }
                }
            }
            debug_assert_eq!(mat.nnz(), nnz);
            mat
        }
    }
}

#[test]
fn test_outer_product() {
    let mut a = sprs::TriMat::new((2, 3));
    a.add_triplet(0, 1, 2);
    a.add_triplet(0, 2, 3);
    a.add_triplet(1, 0, 6);
    a.add_triplet(1, 2, 8);
    let a = a.to_csr();

    let mut b = sprs::TriMat::new((3, 2));
    b.add_triplet(0, 0, 1);
    b.add_triplet(1, 0, 2);
    b.add_triplet(2, 0, 3);
    b.add_triplet(2, 1, -3);
    let b = b.to_csr();

    let c = sparse_sparse_outer_product(a.view(), b.view());
    for (&n, (j, i)) in c.iter() {
        match (j, i) {
            (0, 2) => assert_eq!(n, 2),
            (0, 4) => assert_eq!(n, 3),
            (1, 2) => assert_eq!(n, 4),
            (1, 4) => assert_eq!(n, 6),
            (2, 2) => assert_eq!(n, 6),
            (2, 3) => assert_eq!(n, -6),
            (2, 4) => assert_eq!(n, 9),
            (2, 5) => assert_eq!(n, -9),
            (3, 0) => assert_eq!(n, 6),
            (3, 4) => assert_eq!(n, 8),
            (4, 0) => assert_eq!(n, 12),
            (4, 4) => assert_eq!(n, 16),
            (5, 0) => assert_eq!(n, 18),
            (5, 1) => assert_eq!(n, -18),
            (5, 4) => assert_eq!(n, 24),
            (5, 5) => assert_eq!(n, -24),
            _ => panic!("index ({},{}) should be 0, found {}", j, i, n),
        }
    }
}

#[test]
fn test_outer_product_csc() {
    let mut a = sprs::TriMat::new((2, 3));
    a.add_triplet(0, 1, 2);
    a.add_triplet(0, 2, 3);
    a.add_triplet(1, 0, 6);
    a.add_triplet(1, 2, 8);
    let a = a.to_csc();

    let mut b = sprs::TriMat::new((3, 2));
    b.add_triplet(0, 0, 1);
    b.add_triplet(1, 0, 2);
    b.add_triplet(2, 0, 3);
    b.add_triplet(2, 1, -3);
    let b = b.to_csc();

    let c = sparse_sparse_outer_product(a.view(), b.view());
    for (&n, (j, i)) in c.iter() {
        match (j, i) {
            (0, 2) => assert_eq!(n, 2),
            (0, 4) => assert_eq!(n, 3),
            (1, 2) => assert_eq!(n, 4),
            (1, 4) => assert_eq!(n, 6),
            (2, 2) => assert_eq!(n, 6),
            (2, 3) => assert_eq!(n, -6),
            (2, 4) => assert_eq!(n, 9),
            (2, 5) => assert_eq!(n, -9),
            (3, 0) => assert_eq!(n, 6),
            (3, 4) => assert_eq!(n, 8),
            (4, 0) => assert_eq!(n, 12),
            (4, 4) => assert_eq!(n, 16),
            (5, 0) => assert_eq!(n, 18),
            (5, 1) => assert_eq!(n, -18),
            (5, 4) => assert_eq!(n, 24),
            (5, 5) => assert_eq!(n, -24),
            _ => panic!("index ({},{}) should be 0, found {}", j, i, n),
        }
    }
}

#[test]
fn test_outer_product_2() {
    let mut e0 = sprs::CsMat::zero((10, 1));
    e0.insert(0, 0, 1);
    let mut en = sprs::CsMat::zero((11, 1));
    en.insert(10, 0, 1);

    let v = sparse_sparse_outer_product(e0.view(), en.transpose_view());
    for (&val, (j, i)) in v.iter() {
        match (j, i) {
            (0, 10) => assert_eq!(val, 1),
            _ => panic!("Unexpected element: ({},{}): {}", j, i, val),
        }
    }
}
