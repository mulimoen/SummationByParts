use crate::Float;

#[cfg(feature = "sparse")]
mod jacobi;
#[cfg(feature = "sparse")]
pub use jacobi::*;
#[cfg(feature = "sparse")]
mod kronecker_product;
#[cfg(feature = "sparse")]
pub use kronecker_product::kronecker_product;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, Default)]
pub struct Direction<T> {
    pub north: T,
    pub south: T,
    pub west: T,
    pub east: T,
}

impl<T> Direction<T> {
    pub fn as_ref(&self) -> Direction<&T> {
        Direction {
            north: &self.north,
            south: &self.south,
            west: &self.west,
            east: &self.east,
        }
    }
    pub fn as_mut(&mut self) -> Direction<&mut T> {
        Direction {
            north: &mut self.north,
            south: &mut self.south,
            west: &mut self.west,
            east: &mut self.east,
        }
    }
    pub fn map<U>(self, f: impl Fn(T) -> U) -> Direction<U> {
        Direction {
            north: f(self.north),
            south: f(self.south),
            west: f(self.west),
            east: f(self.east),
        }
    }
}

impl<T> Direction<T> {
    pub fn north(&self) -> &T {
        &self.north
    }
    pub fn north_mut(&mut self) -> &mut T {
        &mut self.north
    }
    pub fn south(&self) -> &T {
        &self.south
    }
    pub fn south_mut(&mut self) -> &mut T {
        &mut self.south
    }
    pub fn east(&self) -> &T {
        &self.east
    }
    pub fn east_mut(&mut self) -> &mut T {
        &mut self.east
    }
    pub fn west(&self) -> &T {
        &self.west
    }
    pub fn west_mut(&mut self) -> &mut T {
        &mut self.west
    }
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
