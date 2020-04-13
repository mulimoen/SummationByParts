use super::InterpolationOperator;
use crate::Float;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1};

mod interpolation4;
pub use interpolation4::Interpolation4;
mod interpolation9;
pub use interpolation9::Interpolation9;
mod interpolation8;
pub use interpolation8::Interpolation8;

mod interpolation9h2;
pub use interpolation9h2::Interpolation9h2;

fn interpolate(
    input: ArrayView1<Float>,
    mut output: ArrayViewMut1<Float>,
    block: ArrayView2<Float>,
    diag: ArrayView2<Float>,
    jump: (usize, usize),
) {
    use ndarray::Axis;

    output.fill(0.0);
    let noutput = output.len();

    for (i, out) in output.iter_mut().enumerate() {
        if i < block.len_of(Axis(0)) {
            for k in 0..block.len_of(Axis(1)) {
                *out += input[k] * block[(i, k)];
            }
        } else if noutput - i <= block.len_of(Axis(0)) {
            let row = noutput - i - 1;
            let index = input.len() - block.len_of(Axis(1));

            for k in 0..block.len_of(Axis(1)) {
                let col = block.len_of(Axis(1)) - k - 1;
                *out += input[index + k] * block[(row, col)];
            }
        } else {
            let n = i - block.len_of(Axis(0));
            let index = jump.0 + jump.1 * (n / diag.len_of(Axis(0)));
            let row = n % diag.len_of(Axis(0));

            for k in 0..diag.len_of(Axis(1)) {
                *out += input[index + k] * diag[(row, k)];
            }
        }
    }
}

#[cfg(test)]
pub(crate) fn test_interpolation_operator<IO: InterpolationOperator>(op: IO, h2: bool) {
    let nc = 101;
    let (x_c, x_f) = if h2 {
        (
            crate::utils::h2linspace(0.0, 1.0, nc),
            crate::utils::h2linspace(0.0, 1.0, 2 * nc - 1),
        )
    } else {
        (
            ndarray::Array1::linspace(0.0, 1.0, 101),
            ndarray::Array1::linspace(0.0, 1.0, 2 * nc - 1),
        )
    };

    let mut ix_f = ndarray::Array1::zeros(x_f.raw_dim());
    op.coarse2fine(x_c.view(), ix_f.view_mut());
    approx::assert_abs_diff_eq!(ix_f, x_f, epsilon = 1e-2);

    let mut ix_c = ndarray::Array1::zeros(x_c.raw_dim());
    op.fine2coarse(x_f.view(), ix_c.view_mut());
    approx::assert_abs_diff_eq!(ix_c, x_c, epsilon = 1e-2);
}
