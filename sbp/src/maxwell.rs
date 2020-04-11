use super::grid::{Grid, Metrics};
use super::integrate;
use super::operators::{SbpOperator, UpwindOperator};
use crate::Float;
use ndarray::azip;
use ndarray::prelude::*;

#[derive(Clone, Debug)]
pub struct Field(pub(crate) Array3<Float>);

impl std::ops::Deref for Field {
    type Target = Array3<Float>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Field {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Field {
    pub fn new(height: usize, width: usize) -> Self {
        let field = Array3::zeros((3, height, width));

        Self(field)
    }

    pub fn nx(&self) -> usize {
        self.0.shape()[2]
    }
    pub fn ny(&self) -> usize {
        self.0.shape()[1]
    }

    pub fn ex(&self) -> ArrayView2<Float> {
        self.slice(s![0, .., ..])
    }
    pub fn hz(&self) -> ArrayView2<Float> {
        self.slice(s![1, .., ..])
    }
    pub fn ey(&self) -> ArrayView2<Float> {
        self.slice(s![2, .., ..])
    }

    pub fn ex_mut(&mut self) -> ArrayViewMut2<Float> {
        self.slice_mut(s![0, .., ..])
    }
    pub fn hz_mut(&mut self) -> ArrayViewMut2<Float> {
        self.slice_mut(s![1, .., ..])
    }
    pub fn ey_mut(&mut self) -> ArrayViewMut2<Float> {
        self.slice_mut(s![2, .., ..])
    }

    pub fn components_mut(
        &mut self,
    ) -> (
        ArrayViewMut2<Float>,
        ArrayViewMut2<Float>,
        ArrayViewMut2<Float>,
    ) {
        let mut iter = self.0.outer_iter_mut();

        let ex = iter.next().unwrap();
        let hz = iter.next().unwrap();
        let ey = iter.next().unwrap();
        assert_eq!(iter.next(), None);

        (ex, hz, ey)
    }
}

#[derive(Debug, Clone)]
pub struct System<SBP: SbpOperator> {
    sys: (Field, Field),
    wb: WorkBuffers,
    grid: Grid,
    metrics: Metrics<SBP>,
}

impl<SBP: SbpOperator> System<SBP> {
    pub fn new(x: Array2<Float>, y: Array2<Float>) -> Self {
        assert_eq!(x.shape(), y.shape());
        let ny = x.shape()[0];
        let nx = x.shape()[1];

        let grid = Grid::new(x, y).unwrap();
        let metrics = grid.metrics().unwrap();

        Self {
            sys: (Field::new(ny, nx), Field::new(ny, nx)),
            grid,
            metrics,
            wb: WorkBuffers::new(ny, nx),
        }
    }

    pub fn field(&self) -> &Field {
        &self.sys.0
    }

    pub fn set_gaussian(&mut self, x0: Float, y0: Float) {
        let (ex, hz, ey) = self.sys.0.components_mut();
        ndarray::azip!(
            (ex in ex, hz in hz, ey in ey,
             &x in &self.grid.x, &y in &self.grid.y)
        {
            *ex = 0.0;
            *ey = 0.0;
            *hz = gaussian(x, x0, y, y0)/32.0;
        });
    }

    pub fn advance(&mut self, dt: Float) {
        fn rhs_adaptor<SBP: SbpOperator>(
            fut: &mut Field,
            prev: &Field,
            _time: Float,
            c: &(&Grid, &Metrics<SBP>),
            m: &mut (Array2<Float>, Array2<Float>, Array2<Float>, Array2<Float>),
        ) {
            let (grid, metrics) = c;
            RHS(fut, prev, grid, metrics, m);
        }
        let mut _time = 0.0;
        integrate::integrate::<integrate::Rk4, _, _, _, _>(
            rhs_adaptor,
            &self.sys.0,
            &mut self.sys.1,
            &mut _time,
            dt,
            &mut self.wb.k,
            &(&self.grid, &self.metrics),
            &mut self.wb.tmp,
        );
        std::mem::swap(&mut self.sys.0, &mut self.sys.1);
    }
}

impl<UO: UpwindOperator> System<UO> {
    /// Using artificial dissipation with the upwind operator
    pub fn advance_upwind(&mut self, dt: Float) {
        fn rhs_adaptor<UO: UpwindOperator>(
            fut: &mut Field,
            prev: &Field,
            _time: Float,
            c: &(&Grid, &Metrics<UO>),
            m: &mut (Array2<Float>, Array2<Float>, Array2<Float>, Array2<Float>),
        ) {
            let (grid, metrics) = c;
            RHS_upwind(fut, prev, grid, metrics, m);
        }
        let mut _time = 0.0;
        integrate::integrate::<integrate::Rk4, _, _, _, _>(
            rhs_adaptor,
            &self.sys.0,
            &mut self.sys.1,
            &mut _time,
            dt,
            &mut self.wb.k,
            &(&self.grid, &self.metrics),
            &mut self.wb.tmp,
        );
        std::mem::swap(&mut self.sys.0, &mut self.sys.1);
    }
}

fn gaussian(x: Float, x0: Float, y: Float, y0: Float) -> Float {
    use crate::consts::PI;

    let x = x - x0;
    let y = y - y0;

    let sigma = 0.05;

    1.0 / (2.0 * PI * sigma * sigma) * (-(x * x + y * y) / (2.0 * sigma * sigma)).exp()
}

#[allow(non_snake_case)]
/// Solving (Au)_x + (Bu)_y
/// with:
///        A               B
///  [ 0,  0,  0]    [ 0,  1,  0]
///  [ 0,  0, -1]    [ 1,  0,  0]
///  [ 0, -1,  0]    [ 0,  0,  0]
///
/// This flux is rotated by the grid metrics
/// (Au)_x + (Bu)_y = 1/J [
///          (J xi_x Au)_xi + (J eta_x Au)_eta
///          (J xi_y Bu)_xi + (J eta_y Bu)_eta
///      ]
/// where J is the grid determinant
///
/// This is used both in fluxes and SAT terms
fn RHS<SBP: SbpOperator>(
    k: &mut Field,
    y: &Field,
    _grid: &Grid,
    metrics: &Metrics<SBP>,
    tmp: &mut (Array2<Float>, Array2<Float>, Array2<Float>, Array2<Float>),
) {
    fluxes(k, y, metrics, tmp);

    let boundaries = BoundaryTerms {
        north: Boundary::This,
        south: Boundary::This,
        west: Boundary::This,
        east: Boundary::This,
    };
    SAT_characteristics(k, y, metrics, &boundaries);

    azip!((k in &mut k.0,
                    &detj in &metrics.detj.broadcast((3, y.ny(), y.nx())).unwrap()) {
        *k /= detj;
    });
}

#[allow(non_snake_case)]
fn RHS_upwind<UO: UpwindOperator>(
    k: &mut Field,
    y: &Field,
    _grid: &Grid,
    metrics: &Metrics<UO>,
    tmp: &mut (Array2<Float>, Array2<Float>, Array2<Float>, Array2<Float>),
) {
    fluxes(k, y, metrics, tmp);
    dissipation(k, y, metrics, tmp);

    let boundaries = BoundaryTerms {
        north: Boundary::This,
        south: Boundary::This,
        west: Boundary::This,
        east: Boundary::This,
    };
    SAT_characteristics(k, y, metrics, &boundaries);

    azip!((k in &mut k.0,
                    &detj in &metrics.detj.broadcast((3, y.ny(), y.nx())).unwrap()) {
        *k /= detj;
    });
}

fn fluxes<SBP: SbpOperator>(
    k: &mut Field,
    y: &Field,
    metrics: &Metrics<SBP>,
    tmp: &mut (Array2<Float>, Array2<Float>, Array2<Float>, Array2<Float>),
) {
    // ex = hz_y
    {
        ndarray::azip!((a in &mut tmp.0,
                        &dxi_dy in &metrics.detj_dxi_dy,
                        &hz in &y.hz())
            *a = dxi_dy * hz
        );
        SBP::diffxi(tmp.0.view(), tmp.1.view_mut());

        ndarray::azip!((b in &mut tmp.2,
                        &deta_dy in &metrics.detj_deta_dy,
                        &hz in &y.hz())
            *b = deta_dy * hz
        );
        SBP::diffeta(tmp.2.view(), tmp.3.view_mut());

        ndarray::azip!((flux in &mut k.ex_mut(), &ax in &tmp.1, &by in &tmp.3)
            *flux = ax + by
        );
    }

    {
        // hz = -ey_x + ex_y
        ndarray::azip!((a in &mut tmp.0,
                        &dxi_dx in &metrics.detj_dxi_dx,
                        &dxi_dy in &metrics.detj_dxi_dy,
                        &ex in &y.ex(),
                        &ey in &y.ey())
            *a = dxi_dx * -ey + dxi_dy * ex
        );
        SBP::diffxi(tmp.0.view(), tmp.1.view_mut());

        ndarray::azip!((b in &mut tmp.2,
                        &deta_dx in &metrics.detj_deta_dx,
                        &deta_dy in &metrics.detj_deta_dy,
                        &ex in &y.ex(),
                        &ey in &y.ey())
            *b = deta_dx * -ey + deta_dy * ex
        );
        SBP::diffeta(tmp.2.view(), tmp.3.view_mut());

        ndarray::azip!((flux in &mut k.hz_mut(), &ax in &tmp.1, &by in &tmp.3)
            *flux = ax + by
        );
    }

    // ey = -hz_x
    {
        ndarray::azip!((a in &mut tmp.0,
                        &dxi_dx in &metrics.detj_dxi_dx,
                        &hz in &y.hz())
            *a = dxi_dx * -hz
        );
        SBP::diffxi(tmp.0.view(), tmp.1.view_mut());

        azip!((b in &mut tmp.2,
                        &deta_dx in &metrics.detj_deta_dx,
                        &hz in &y.hz())
            *b = deta_dx * -hz
        );
        SBP::diffeta(tmp.2.view(), tmp.3.view_mut());

        azip!((flux in &mut k.ey_mut(), &ax in &tmp.1, &by in &tmp.3)
            *flux = ax + by
        );
    }
}

fn dissipation<UO: UpwindOperator>(
    k: &mut Field,
    y: &Field,
    metrics: &Metrics<UO>,
    tmp: &mut (Array2<Float>, Array2<Float>, Array2<Float>, Array2<Float>),
) {
    // ex component
    {
        ndarray::azip!((a in &mut tmp.0,
                        &kx in &metrics.detj_dxi_dx,
                        &ky in &metrics.detj_dxi_dy,
                        &ex in &y.ex(),
                        &ey in &y.ey()) {
            let r = Float::hypot(kx, ky);
            *a = ky*ky/r * ex + -kx*ky/r*ey;
        });
        UO::dissxi(tmp.0.view(), tmp.1.view_mut());

        ndarray::azip!((b in &mut tmp.2,
                    &kx in &metrics.detj_deta_dx,
                    &ky in &metrics.detj_deta_dy,
                    &ex in &y.ex(),
                    &ey in &y.ey()) {
            let r = Float::hypot(kx, ky);
            *b = ky*ky/r * ex + -kx*ky/r*ey;
        });
        UO::disseta(tmp.2.view(), tmp.3.view_mut());

        ndarray::azip!((flux in &mut k.ex_mut(), &ax in &tmp.1, &by in &tmp.3)
            *flux += ax + by
        );
    }

    // hz component
    {
        ndarray::azip!((a in &mut tmp.0,
                        &kx in &metrics.detj_dxi_dx,
                        &ky in &metrics.detj_dxi_dy,
                        &hz in &y.hz()) {
            let r = Float::hypot(kx, ky);
            *a = r * hz;
        });
        UO::dissxi(tmp.0.view(), tmp.1.view_mut());

        ndarray::azip!((b in &mut tmp.2,
                        &kx in &metrics.detj_deta_dx,
                        &ky in &metrics.detj_deta_dy,
                        &hz in &y.hz()) {
            let r = Float::hypot(kx, ky);
            *b = r * hz;
        });
        UO::disseta(tmp.2.view(), tmp.3.view_mut());

        ndarray::azip!((flux in &mut k.hz_mut(), &ax in &tmp.1, &by in &tmp.3)
            *flux += ax + by
        );
    }

    // ey
    {
        ndarray::azip!((a in &mut tmp.0,
                        &kx in &metrics.detj_dxi_dx,
                        &ky in &metrics.detj_dxi_dy,
                        &ex in &y.ex(),
                        &ey in &y.ey()) {
            let r = Float::hypot(kx, ky);
            *a = -kx*ky/r * ex + kx*kx/r*ey;
        });
        UO::dissxi(tmp.0.view(), tmp.1.view_mut());

        ndarray::azip!((b in &mut tmp.2,
                    &kx in &metrics.detj_deta_dx,
                    &ky in &metrics.detj_deta_dy,
                    &ex in &y.ex(),
                    &ey in &y.ey()) {
            let r = Float::hypot(kx, ky);
            *b = -kx*ky/r * ex + kx*kx/r*ey;
        });
        UO::disseta(tmp.2.view(), tmp.3.view_mut());

        ndarray::azip!((flux in &mut k.hz_mut(), &ax in &tmp.1, &by in &tmp.3)
            *flux += ax + by
        );
    }
}

#[derive(Clone, Debug)]
pub enum Boundary {
    This,
}

#[derive(Clone, Debug)]
pub struct BoundaryTerms {
    pub north: Boundary,
    pub south: Boundary,
    pub east: Boundary,
    pub west: Boundary,
}

#[allow(non_snake_case)]
/// Boundary conditions (SAT)
fn SAT_characteristics<SBP: SbpOperator>(
    k: &mut Field,
    y: &Field,
    metrics: &Metrics<SBP>,
    boundaries: &BoundaryTerms,
) {
    let ny = y.ny();
    let nx = y.nx();

    fn positive_flux(kx: Float, ky: Float) -> [[Float; 3]; 3] {
        let r = (kx * kx + ky * ky).sqrt();
        [
            [ky * ky / r / 2.0, ky / 2.0, -kx * ky / r / 2.0],
            [ky / 2.0, r / 2.0, -kx / 2.0],
            [-kx * ky / r / 2.0, -kx / 2.0, kx * kx / r / 2.0],
        ]
    }
    fn negative_flux(kx: Float, ky: Float) -> [[Float; 3]; 3] {
        let r = (kx * kx + ky * ky).sqrt();
        [
            [-ky * ky / r / 2.0, ky / 2.0, kx * ky / r / 2.0],
            [ky / 2.0, -r / 2.0, -kx / 2.0],
            [kx * ky / r / 2.0, -kx / 2.0, -kx * kx / r / 2.0],
        ]
    }

    {
        let g = match boundaries.east {
            Boundary::This => y.slice(s![.., .., 0]),
        };
        // East boundary
        let hinv = 1.0 / (SBP::h()[0] / (nx - 1) as Float);
        for ((((mut k, v), g), &kx), &ky) in k
            .slice_mut(s![.., .., nx - 1])
            .gencolumns_mut()
            .into_iter()
            .zip(y.slice(s![.., .., nx - 1]).gencolumns())
            .zip(g.gencolumns())
            .zip(metrics.detj_dxi_dx.slice(s![.., nx - 1]))
            .zip(metrics.detj_dxi_dy.slice(s![.., nx - 1]))
        {
            // East boundary, positive flux
            let tau = -1.0;

            let v = (v[0], v[1], v[2]);
            let g = (g[0], g[1], g[2]);

            let plus = positive_flux(kx, ky);

            k[0] += tau
                * hinv
                * (plus[0][0] * (v.0 - g.0) + plus[0][1] * (v.1 - g.1) + plus[0][2] * (v.2 - g.2));
            k[1] += tau
                * hinv
                * (plus[1][0] * (v.0 - g.0) + plus[1][1] * (v.1 - g.1) + plus[1][2] * (v.2 - g.2));
            k[2] += tau
                * hinv
                * (plus[2][0] * (v.0 - g.0) + plus[2][1] * (v.1 - g.1) + plus[2][2] * (v.2 - g.2));
        }
    }
    {
        // West boundary, negative flux
        let g = match boundaries.east {
            Boundary::This => y.slice(s![.., .., nx - 1]),
        };
        let hinv = 1.0 / (SBP::h()[0] / (nx - 1) as Float);
        for ((((mut k, v), g), &kx), &ky) in k
            .slice_mut(s![.., .., 0])
            .gencolumns_mut()
            .into_iter()
            .zip(y.slice(s![.., .., 0]).gencolumns())
            .zip(g.gencolumns())
            .zip(metrics.detj_dxi_dx.slice(s![.., 0]))
            .zip(metrics.detj_dxi_dy.slice(s![.., 0]))
        {
            let tau = 1.0;

            let v = (v[0], v[1], v[2]);
            let g = (g[0], g[1], g[2]);

            let minus = negative_flux(kx, ky);

            k[0] += tau
                * hinv
                * (minus[0][0] * (v.0 - g.0)
                    + minus[0][1] * (v.1 - g.1)
                    + minus[0][2] * (v.2 - g.2));
            k[1] += tau
                * hinv
                * (minus[1][0] * (v.0 - g.0)
                    + minus[1][1] * (v.1 - g.1)
                    + minus[1][2] * (v.2 - g.2));
            k[2] += tau
                * hinv
                * (minus[2][0] * (v.0 - g.0)
                    + minus[2][1] * (v.1 - g.1)
                    + minus[2][2] * (v.2 - g.2));
        }
    }

    {
        let g = match boundaries.north {
            Boundary::This => y.slice(s![.., 0, ..]),
        };
        let hinv = 1.0 / (SBP::h()[0] / (ny - 1) as Float);
        for ((((mut k, v), g), &kx), &ky) in k
            .slice_mut(s![.., ny - 1, ..])
            .gencolumns_mut()
            .into_iter()
            .zip(y.slice(s![.., ny - 1, ..]).gencolumns())
            .zip(g.gencolumns())
            .zip(metrics.detj_deta_dx.slice(s![ny - 1, ..]))
            .zip(metrics.detj_deta_dy.slice(s![ny - 1, ..]))
        {
            // North boundary, positive flux
            let tau = -1.0;
            let v = (v[0], v[1], v[2]);
            let g = (g[0], g[1], g[2]);

            let plus = positive_flux(kx, ky);

            k[0] += tau
                * hinv
                * (plus[0][0] * (v.0 - g.0) + plus[0][1] * (v.1 - g.1) + plus[0][2] * (v.2 - g.2));
            k[1] += tau
                * hinv
                * (plus[1][0] * (v.0 - g.0) + plus[1][1] * (v.1 - g.1) + plus[1][2] * (v.2 - g.2));
            k[2] += tau
                * hinv
                * (plus[2][0] * (v.0 - g.0) + plus[2][1] * (v.1 - g.1) + plus[2][2] * (v.2 - g.2));
        }
    }

    {
        let g = match boundaries.south {
            Boundary::This => y.slice(s![.., ny - 1, ..]),
        };
        let hinv = 1.0 / (SBP::h()[0] / (ny - 1) as Float);
        for ((((mut k, v), g), &kx), &ky) in k
            .slice_mut(s![.., 0, ..])
            .gencolumns_mut()
            .into_iter()
            .zip(y.slice(s![.., 0, ..]).gencolumns())
            .zip(g.gencolumns())
            .zip(metrics.detj_deta_dx.slice(s![0, ..]))
            .zip(metrics.detj_deta_dy.slice(s![0, ..]))
        {
            // South boundary, negative flux

            let tau = 1.0;
            let v = (v[0], v[1], v[2]);
            let g = (g[0], g[1], g[2]);

            let minus = negative_flux(kx, ky);

            k[0] += tau
                * hinv
                * (minus[0][0] * (v.0 - g.0)
                    + minus[0][1] * (v.1 - g.1)
                    + minus[0][2] * (v.2 - g.2));
            k[1] += tau
                * hinv
                * (minus[1][0] * (v.0 - g.0)
                    + minus[1][1] * (v.1 - g.1)
                    + minus[1][2] * (v.2 - g.2));
            k[2] += tau
                * hinv
                * (minus[2][0] * (v.0 - g.0)
                    + minus[2][1] * (v.1 - g.1)
                    + minus[2][2] * (v.2 - g.2));
        }
    }
}

#[derive(Clone, Debug)]
pub struct WorkBuffers {
    k: [Field; 4],
    tmp: (Array2<Float>, Array2<Float>, Array2<Float>, Array2<Float>),
}

impl WorkBuffers {
    pub fn new(ny: usize, nx: usize) -> Self {
        let arr2 = Array2::zeros((ny, nx));
        let arr3 = Field::new(ny, nx);
        Self {
            k: [arr3.clone(), arr3.clone(), arr3.clone(), arr3],
            tmp: (arr2.clone(), arr2.clone(), arr2.clone(), arr2),
        }
    }
}
