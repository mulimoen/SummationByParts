use ndarray::Array2;
use sbp::operators::{self, SbpOperator2d};
use sbp::Float;

const W: usize = 64;
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

iai::main!(
    baseline,
    upwind4_diffxi,
    upwind9_diffxi,
    trad4_diffxi,
    trad8_diffxi,
    upwind4_diffeta,
    upwind9_diffeta,
    trad4_diffeta,
    trad8_diffeta
);
