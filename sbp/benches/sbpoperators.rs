use ndarray::Array2;
use sbp::operators::{self, SbpOperator2d};
use sbp::Float;

const W: usize = 128;
const H: usize = 64;

fn baseline() {
    let _x = Array2::<Float>::from_shape_fn((W, H), |(j, i)| (j * W + i) as Float);
    let _res = Array2::<Float>::zeros((W, H));
}

fn upwind4_diffxi() {
    let x = Array2::from_shape_fn((W, H), |(j, i)| (j * W + i) as Float);
    let mut res = Array2::zeros((W, H));

    operators::Upwind4.diffxi(x.view(), res.view_mut());
}
fn upwind9_diffxi() {
    let x = Array2::from_shape_fn((W, H), |(j, i)| (j * W + i) as Float);
    let mut res = Array2::zeros((W, H));

    operators::Upwind9.diffxi(x.view(), res.view_mut());
}
fn trad4_diffxi() {
    let x = Array2::from_shape_fn((W, H), |(j, i)| (j * W + i) as Float);
    let mut res = Array2::zeros((W, H));

    operators::SBP4.diffxi(x.view(), res.view_mut());
}
fn trad8_diffxi() {
    let x = Array2::from_shape_fn((W, H), |(j, i)| (j * W + i) as Float);
    let mut res = Array2::zeros((W, H));

    operators::SBP8.diffxi(x.view(), res.view_mut());
}
fn upwind4_diffeta() {
    let x = Array2::from_shape_fn((W, H), |(j, i)| (j * W + i) as Float);
    let mut res = Array2::zeros((W, H));

    operators::Upwind4.diffeta(x.view(), res.view_mut());
}
fn upwind9_diffeta() {
    let x = Array2::from_shape_fn((W, H), |(j, i)| (j * W + i) as Float);
    let mut res = Array2::zeros((W, H));

    operators::Upwind9.diffeta(x.view(), res.view_mut());
}
fn trad4_diffeta() {
    let x = Array2::from_shape_fn((W, H), |(j, i)| (j * W + i) as Float);
    let mut res = Array2::zeros((W, H));

    operators::SBP4.diffeta(x.view(), res.view_mut());
}
fn trad8_diffeta() {
    let x = Array2::from_shape_fn((W, H), |(j, i)| (j * W + i) as Float);
    let mut res = Array2::zeros((W, H));

    operators::SBP8.diffeta(x.view(), res.view_mut());
}

#[cfg(feature = "sparse")]
fn baseline_sparse() {
    let dx = operators::Upwind4.op_xi().diff_matrix(W);
    let _dx = sprs::kronecker_product(sprs::CsMat::eye(H).view(), dx.view());

    let _x = Array2::<Float>::from_shape_fn((W, H), |(j, i)| (j * W + i) as Float);
    let _res = Array2::<Float>::zeros((W, H));
}

#[cfg(feature = "sparse")]
fn upwind4_diffxi_sparse() {
    let dx = operators::Upwind4.op_xi().diff_matrix(W);
    let dx = sprs::kronecker_product(sprs::CsMat::eye(H).view(), dx.view());

    let x = Array2::from_shape_fn((W, H), |(j, i)| (j * W + i) as Float);
    let mut res = Array2::zeros((W, H));

    sprs::prod::mul_acc_mat_vec_csr(
        dx.view(),
        x.as_slice().unwrap(),
        res.as_slice_mut().unwrap(),
    );
}

#[cfg(not(feature = "sparse"))]
iai::main!(
    baseline,
    upwind4_diffxi,
    upwind9_diffxi,
    trad4_diffxi,
    trad8_diffxi,
    upwind4_diffeta,
    upwind9_diffeta,
    trad4_diffeta,
    trad8_diffeta,
);

#[cfg(feature = "sparse")]
iai::main!(
    baseline,
    upwind4_diffxi,
    upwind9_diffxi,
    trad4_diffxi,
    trad8_diffxi,
    upwind4_diffeta,
    upwind9_diffeta,
    trad4_diffeta,
    trad8_diffeta,
    baseline_sparse,
    upwind4_diffxi_sparse,
);
