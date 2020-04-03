use crate::Float;
use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct Grid {
    pub(crate) x: Array2<Float>,
    pub(crate) y: Array2<Float>,
}

#[derive(Debug, Clone)]
pub struct Metrics<SBP>
where
    SBP: super::operators::SbpOperator,
{
    pub(crate) detj: Array2<Float>,
    pub(crate) detj_dxi_dx: Array2<Float>,
    pub(crate) detj_dxi_dy: Array2<Float>,
    pub(crate) detj_deta_dx: Array2<Float>,
    pub(crate) detj_deta_dy: Array2<Float>,

    operator: std::marker::PhantomData<SBP>,
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

    pub fn metrics<SBP: super::operators::SbpOperator>(
        &self,
    ) -> Result<Metrics<SBP>, ndarray::ShapeError> {
        Metrics::new(self)
    }
}

impl<SBP: super::operators::SbpOperator> Metrics<SBP> {
    fn new(grid: &Grid) -> Result<Self, ndarray::ShapeError> {
        let ny = grid.ny();
        let nx = grid.nx();
        let x = &grid.x;
        let y = &grid.y;

        let mut dx_dxi = Array2::zeros((ny, nx));
        SBP::diffxi(x.view(), dx_dxi.view_mut());
        let mut dx_deta = Array2::zeros((ny, nx));
        SBP::diffeta(x.view(), dx_deta.view_mut());
        let mut dy_dxi = Array2::zeros((ny, nx));
        SBP::diffxi(y.view(), dy_dxi.view_mut());
        let mut dy_deta = Array2::zeros((ny, nx));
        SBP::diffeta(y.view(), dy_deta.view_mut());

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
            operator: std::marker::PhantomData,
        })
    }
}
