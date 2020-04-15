use crate::Float;

pub struct Direction<T> {
    pub north: T,
    pub south: T,
    pub west: T,
    pub east: T,
}

pub fn h2linspace(start: Float, end: Float, n: usize) -> ndarray::Array1<Float> {
    let h = (end - start) / (n - 2) as Float;
    ndarray::Array1::from_shape_fn(n, |i| match i {
        0 => start,
        i if i == n - 1 => end,
        i => start + h * (i as Float - 0.5),
    })
}

#[test]
fn test_h2linspace() {
    let x = h2linspace(-1.0, 1.0, 50);
    println!("{}", x);
    approx::assert_abs_diff_eq!(x[0], -1.0, epsilon = 1e-6);
    approx::assert_abs_diff_eq!(x[49], 1.0, epsilon = 1e-6);
    let hend = x[1] - x[0];
    let h = x[2] - x[1];
    approx::assert_abs_diff_eq!(x[49] - x[48], hend, epsilon = 1e-6);
    approx::assert_abs_diff_eq!(2.0 * hend, h, epsilon = 1e-6);
    for i in 1..48 {
        approx::assert_abs_diff_eq!(x[i + 1] - x[i], h, epsilon = 1e-6);
    }
}
