use super::operators::SbpOperator2d;
use crate::Float;
use ndarray::{Array2, ArrayView2};

#[derive(Debug, Clone)]
pub struct Grid {
    pub(crate) x: Array2<Float>,
    pub(crate) y: Array2<Float>,
}

#[derive(Debug, Clone)]
pub struct Metrics {
    pub(crate) detj: Array2<Float>,
    pub(crate) detj_dxi_dx: Array2<Float>,
    pub(crate) detj_dxi_dy: Array2<Float>,
    pub(crate) detj_deta_dx: Array2<Float>,
    pub(crate) detj_deta_dy: Array2<Float>,
}

impl Grid {
    pub fn new(x: Array2<Float>, y: Array2<Float>) -> Result<Self, ndarray::ShapeError> {
        assert_eq!(x.shape(), y.shape());

        Ok(Self { x, y })
    }
    pub fn nx(&self) -> usize {
        self.x.shape()[1]
    }
    pub fn ny(&self) -> usize {
        self.x.shape()[0]
    }

    pub fn x(&self) -> ndarray::ArrayView2<Float> {
        self.x.view()
    }
    pub fn y(&self) -> ndarray::ArrayView2<Float> {
        self.y.view()
    }

    pub fn metrics(&self, op: &dyn SbpOperator2d) -> Result<Metrics, ndarray::ShapeError> {
        Metrics::new(self, op)
    }

    pub fn north(&self) -> (ndarray::ArrayView1<Float>, ndarray::ArrayView1<Float>) {
        (
            self.y.slice(ndarray::s![self.ny() - 1, ..]),
            self.x.slice(ndarray::s![self.ny() - 1, ..]),
        )
    }
    pub fn south(&self) -> (ndarray::ArrayView1<Float>, ndarray::ArrayView1<Float>) {
        (
            self.y.slice(ndarray::s![0, ..]),
            self.x.slice(ndarray::s![0, ..]),
        )
    }
    pub fn west(&self) -> (ndarray::ArrayView1<Float>, ndarray::ArrayView1<Float>) {
        (
            self.y.slice(ndarray::s![.., 0]),
            self.x.slice(ndarray::s![.., 0]),
        )
    }
    pub fn east(&self) -> (ndarray::ArrayView1<Float>, ndarray::ArrayView1<Float>) {
        (
            self.y.slice(ndarray::s![.., self.nx() - 1]),
            self.x.slice(ndarray::s![.., self.nx() - 1]),
        )
    }
}

impl Metrics {
    fn new(grid: &Grid, op: &dyn SbpOperator2d) -> Result<Self, ndarray::ShapeError> {
        let ny = grid.ny();
        let nx = grid.nx();
        let x = &grid.x;
        let y = &grid.y;

        let mut dx_dxi = Array2::zeros((ny, nx));
        op.diffxi(x.view(), dx_dxi.view_mut());
        let mut dx_deta = Array2::zeros((ny, nx));
        op.diffeta(x.view(), dx_deta.view_mut());
        let mut dy_dxi = Array2::zeros((ny, nx));
        op.diffxi(y.view(), dy_dxi.view_mut());
        let mut dy_deta = Array2::zeros((ny, nx));
        op.diffeta(y.view(), dy_deta.view_mut());

        let mut detj = Array2::zeros((ny, nx));
        ndarray::azip!((detj in &mut detj,
                        &dx_dxi in &dx_dxi,
                        &dx_deta in &dx_deta,
                        &dy_dxi in &dy_dxi,
                        &dy_deta in &dy_deta) {
            *detj = dx_dxi * dy_deta - dx_deta * dy_dxi;
            assert!(*detj > 0.0);
        });

        let detj_dxi_dx = dy_deta;
        let detj_dxi_dy = {
            let mut dx_deta = dx_deta;
            dx_deta.mapv_inplace(|v| -v);
            dx_deta
        };
        let detj_deta_dx = {
            let mut dy_dxi = dy_dxi;
            dy_dxi.mapv_inplace(|v| -v);
            dy_dxi
        };
        let detj_deta_dy = dx_dxi;

        Ok(Self {
            detj,
            detj_dxi_dx,
            detj_dxi_dy,
            detj_deta_dx,
            detj_deta_dy,
        })
    }
}

impl Metrics {
    pub fn detj(&self) -> ArrayView2<Float> {
        self.detj.view()
    }
    pub fn detj_dxi_dx(&self) -> ArrayView2<Float> {
        self.detj_dxi_dx.view()
    }
    pub fn detj_dxi_dy(&self) -> ArrayView2<Float> {
        self.detj_dxi_dy.view()
    }
    pub fn detj_deta_dx(&self) -> ArrayView2<Float> {
        self.detj_deta_dx.view()
    }
    pub fn detj_deta_dy(&self) -> ArrayView2<Float> {
        self.detj_deta_dy.view()
    }
}
