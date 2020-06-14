/// Computes the sparse kronecker product
/// M = A \kron B
#[allow(non_snake_case)]
#[must_use]
pub fn sparse_sparse_outer_product<
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
