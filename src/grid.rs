use ndarray::Array2;
pub struct Grid<SBP>
where
    SBP: super::operators::SbpOperator,
{
    pub(crate) x: Array2<f32>,
    pub(crate) y: Array2<f32>,

    pub(crate) detj: Array2<f32>,
    pub(crate) detj_dxi_dx: Array2<f32>,
    pub(crate) detj_dxi_dy: Array2<f32>,
    pub(crate) detj_deta_dx: Array2<f32>,
    pub(crate) detj_deta_dy: Array2<f32>,

    operator: std::marker::PhantomData<SBP>,
}

impl<SBP: super::operators::SbpOperator> Grid<SBP> {
    pub fn new(nx: usize, ny: usize, x: &[f32], y: &[f32]) -> Result<Self, ndarray::ShapeError> {
        let x = Array2::from_shape_vec((ny, nx), x.to_vec())?;
        let y = Array2::from_shape_vec((ny, nx), y.to_vec())?;

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
            x,
            y,
            detj,
            detj_dxi_dx,
            detj_dxi_dy,
            detj_deta_dx,
            detj_deta_dy,
            operator: std::marker::PhantomData,
        })
    }
}
