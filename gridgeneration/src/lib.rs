use ndarray::Array2;

pub use sbp::{consts::*, grid::Grid, Float};

pub fn unit(ny: usize, nx: usize) -> Grid {
    let x = Array2::from_shape_fn((ny, nx), |(_j, i)| i as Float / (nx - 1) as Float);
    let y = Array2::from_shape_fn((ny, nx), |(j, _i)| j as Float / (ny - 1) as Float);
    Grid::new(x, y).unwrap()
}

pub fn shell(ny: usize, nx: usize, mut inner: Float, mut outer: Float) -> Grid {
    let mut unit = unit(ny, nx);

    #[allow(unused)]
    enum Mode {
        Log,
        Exp,
        Normal,
    }

    let mode = Mode::Log;
    match mode {
        Mode::Exp => {
            inner = inner.exp();
            outer = outer.exp();
        }
        Mode::Log => {
            inner = inner.ln();
            outer = outer.ln();
        }
        _ => {}
    }

    let (mut gx, mut gy) = unit.components();
    gx.iter_mut().zip(gy.iter_mut()).for_each(|(x, y)| {
        let mut r = (*x) * (outer - inner) + inner;
        match mode {
            Mode::Exp => {
                r = r.ln();
            }
            Mode::Log => {
                r = r.exp();
            }
            _ => {}
        }
        let theta = *y * 2.0 * PI;

        *x = r * theta.cos();
        *y = r * theta.sin();
    });

    unit
}

pub fn vertical_interpolation(jlen: usize, north: &[Float], south: &[Float]) -> Grid {
    let ilen = north.len();
    assert_eq!(north.len(), south.len());
    let mut grid = unit(jlen, ilen);

    for j in 0..jlen {
        for i in 0..ilen {
            let y = grid.y()[(j, i)];
            grid.y_mut()[(j, i)] = (north[i] - south[i]) * y + south[i];
        }
    }
    grid
}

pub fn horizontal_interpolation(ilen: usize, east: &[Float], west: &[Float]) -> Grid {
    let jlen = east.len();
    assert_eq!(east.len(), west.len());
    let mut grid = unit(jlen, ilen);

    for j in 0..jlen {
        for i in 0..ilen {
            let x = grid.x()[(j, i)];
            grid.x_mut()[(j, i)] = (east[j] - west[j]) * x + west[j];
        }
    }
    grid
}

pub fn horizontal_interpolation_dual(
    ilen: usize,
    east: (&[Float], &[Float]),
    west: (&[Float], &[Float]),
) -> Grid {
    let jlen = east.0.len();
    assert_eq!(east.0.len(), east.1.len());
    assert_eq!(east.0.len(), west.0.len());
    assert_eq!(east.0.len(), west.1.len());
    let mut grid = unit(jlen, ilen);

    for j in 0..jlen {
        for i in 0..ilen {
            let x = grid.x()[(j, i)];
            grid.x_mut()[(j, i)] = (east.1[j] - west.1[j]) * x + west.1[j];
            grid.y_mut()[(j, i)] = (east.0[j] - west.0[j]) * x + west.0[j];
        }
    }
    grid
}

/// [east] : fn([0, 1]) -> (y, x)
/// [west] : fn([0, 1]) -> (y, x)
pub fn horizontal_interpolation_dual2(
    ilen: usize,
    jlen: usize,
    east: impl Fn(Float) -> (Float, Float),
    west: impl Fn(Float) -> (Float, Float),
) -> Grid {
    let mut grid = unit(jlen, ilen);

    for j in 0..jlen {
        for i in 0..ilen {
            let x = grid.x()[(j, i)];
            let y = grid.y()[(j, i)];
            let east = east(y);
            let west = west(y);
            grid.x_mut()[(j, i)] = (east.1 - west.1) * x + west.1;
            grid.y_mut()[(j, i)] = (east.0 - west.0) * x + west.0;
        }
    }
    grid
}
