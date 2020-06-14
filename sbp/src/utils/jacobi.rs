use crate::Float;

/// A x = b
/// with A and b known
/// x should contain a first guess of
pub fn jacobi_method(
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

pub fn jacobi_step(a: sprs::CsMatView<Float>, b: &[Float], x0: &[Float], x: &mut [Float]) {
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
