use ndarray::azip;
use ndarray::prelude::*;
use sbp::grid::{Grid, Metrics};
use sbp::operators::{SbpOperator2d, UpwindOperator2d};
use sbp::Float;

#[cfg(feature = "sparse")]
pub mod sparse;

#[derive(Clone, Debug)]
pub struct Field(pub(crate) Array3<Float>);

impl integrate::Integrable for Field {
    type State = Field;
    type Diff = Field;

    fn assign(s: &mut Self::State, o: &Self::State) {
        s.0.assign(&o.0);
    }
    fn scaled_add(s: &mut Self::State, o: &Self::Diff, scale: Float) {
        s.0.scaled_add(scale, &o.0);
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

    pub(crate) fn slice<Do: Dimension>(
        &self,
        info: &ndarray::SliceInfo<[ndarray::SliceOrIndex; 3], Do>,
    ) -> ArrayView<Float, Do> {
        self.0.slice(info)
    }

    pub(crate) fn slice_mut<Do: Dimension>(
        &mut self,
        info: &ndarray::SliceInfo<[ndarray::SliceOrIndex; 3], Do>,
    ) -> ArrayViewMut<Float, Do> {
        self.0.slice_mut(info)
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
        self.0
            .multi_slice_mut((s![0, .., ..], s![1, .., ..], s![2, .., ..]))
    }
}

#[derive(Debug, Clone)]
pub struct System<SBP: SbpOperator2d> {
    sys: (Field, Field),
    wb: WorkBuffers,
    grid: Grid,
    metrics: Metrics,
    op: SBP,
    #[cfg(feature = "sparse")]
    rhs: sprs::CsMat<Float>,
    #[cfg(feature = "sparse")]
    lhs: sprs::CsMat<Float>,
}

impl<SBP: SbpOperator2d> System<SBP> {
    pub fn new(x: Array2<Float>, y: Array2<Float>, op: SBP) -> Self {
        assert_eq!(x.shape(), y.shape());
        let ny = x.shape()[0];
        let nx = x.shape()[1];

        let grid = Grid::new(x, y).unwrap();
        let metrics = grid.metrics(&op).unwrap();

        #[cfg(feature = "sparse")]
        let rhs = sparse::rhs_matrix(&op, &grid).rhs;

        #[cfg(feature = "sparse")]
        let lhs = sparse::implicit_matrix(rhs.view(), 0.2 / std::cmp::max(ny, nx) as Float);

        Self {
            op,
            sys: (Field::new(ny, nx), Field::new(ny, nx)),
            grid,
            metrics,
            wb: WorkBuffers::new(ny, nx),
            #[cfg(feature = "sparse")]
            rhs,
            #[cfg(feature = "sparse")]
            lhs,
        }
    }

    pub fn field(&self) -> &Field {
        &self.sys.0
    }

    pub fn set_gaussian(&mut self, x0: Float, y0: Float) {
        let (ex, hz, ey) = self.sys.0.components_mut();
        ndarray::azip!(
            (ex in ex, hz in hz, ey in ey,
             &x in &self.grid.x(), &y in &self.grid.y())
        {
            *ex = 0.0;
            *ey = 0.0;
            *hz = gaussian(x, x0, y, y0)/32.0;
        });
    }

    pub fn advance(&mut self, dt: Float) {
        let op = &self.op;
        let grid = &self.grid;
        let metrics = &self.metrics;
        let wb = &mut self.wb.tmp;
        let rhs_adaptor = move |fut: &mut Field, prev: &Field, _time: Float| {
            RHS(op, fut, prev, grid, metrics, wb);
        };
        let mut _time = 0.0;
        integrate::integrate::<integrate::Rk4, Field, _>(
            rhs_adaptor,
            &self.sys.0,
            &mut self.sys.1,
            &mut _time,
            dt,
            &mut self.wb.k,
        );
        std::mem::swap(&mut self.sys.0, &mut self.sys.1);
    }
    #[cfg(feature = "sparse")]
    pub fn advance_sparse(&mut self, dt: Float) {
        let rhs = self.rhs.view();
        //let lhs = self.explicit.view();
        let rhs_f = |next: &mut Field, now: &Field, _t: Float| {
            next.0.fill(0.0);
            sprs::prod::mul_acc_mat_vec_csr(
                rhs,
                now.0.as_slice().unwrap(),
                next.0.as_slice_mut().unwrap(),
            );
            // sprs::lingalg::dsolve(..)
        };
        integrate::integrate::<integrate::Rk4, Field, _>(
            rhs_f,
            &self.sys.0,
            &mut self.sys.1,
            &mut 0.0,
            dt,
            &mut self.wb.k[..],
        );
        std::mem::swap(&mut self.sys.0, &mut self.sys.1);
    }
    #[cfg(feature = "sparse")]
    pub fn advance_implicit(&mut self) {
        let lhs = self.lhs.view();

        let b = self.sys.0.clone();

        sbp::utils::jacobi_method(
            lhs,
            b.0.as_slice().unwrap(),
            self.sys.0 .0.as_slice_mut().unwrap(),
            self.sys.1 .0.as_slice_mut().unwrap(),
            10,
        );
    }
}

impl<UO: SbpOperator2d + UpwindOperator2d> System<UO> {
    /// Using artificial dissipation with the upwind operator
    pub fn advance_upwind(&mut self, dt: Float) {
        let op = &self.op;
        let grid = &self.grid;
        let metrics = &self.metrics;
        let wb = &mut self.wb.tmp;
        let rhs_adaptor = move |fut: &mut Field, prev: &Field, _time: Float| {
            RHS_upwind(op, fut, prev, grid, metrics, wb);
        };
        let mut _time = 0.0;
        integrate::integrate::<integrate::Rk4, Field, _>(
            rhs_adaptor,
            &self.sys.0,
            &mut self.sys.1,
            &mut _time,
            dt,
            &mut self.wb.k,
        );
        std::mem::swap(&mut self.sys.0, &mut self.sys.1);
    }
}

fn gaussian(x: Float, x0: Float, y: Float, y0: Float) -> Float {
    use sbp::consts::PI;

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
fn RHS<SBP: SbpOperator2d>(
    op: &SBP,
    k: &mut Field,
    y: &Field,
    _grid: &Grid,
    metrics: &Metrics,
    tmp: &mut (Array2<Float>, Array2<Float>, Array2<Float>, Array2<Float>),
) {
    fluxes(op, k, y, metrics, tmp);

    let boundaries = BoundaryTerms {
        north: Boundary::This,
        south: Boundary::This,
        west: Boundary::This,
        east: Boundary::This,
    };
    SAT_characteristics(op, k, y, metrics, &boundaries);

    azip!((k in &mut k.0,
                    &detj in &metrics.detj().broadcast((3, y.ny(), y.nx())).unwrap()) {
        *k /= detj;
    });
}

#[allow(non_snake_case)]
fn RHS_upwind<UO: SbpOperator2d + UpwindOperator2d>(
    op: &UO,
    k: &mut Field,
    y: &Field,
    _grid: &Grid,
    metrics: &Metrics,
    tmp: &mut (Array2<Float>, Array2<Float>, Array2<Float>, Array2<Float>),
) {
    fluxes(op, k, y, metrics, tmp);
    dissipation(op, k, y, metrics, tmp);

    let boundaries = BoundaryTerms {
        north: Boundary::This,
        south: Boundary::This,
        west: Boundary::This,
        east: Boundary::This,
    };
    SAT_characteristics(op, k, y, metrics, &boundaries);

    azip!((k in &mut k.0,
                    &detj in &metrics.detj().broadcast((3, y.ny(), y.nx())).unwrap()) {
        *k /= detj;
    });
}

fn fluxes<SBP: sbp::operators::SbpOperator2d>(
    op: &SBP,
    k: &mut Field,
    y: &Field,
    metrics: &Metrics,
    tmp: &mut (Array2<Float>, Array2<Float>, Array2<Float>, Array2<Float>),
) {
    // ex = hz_y
    {
        ndarray::azip!((a in &mut tmp.0,
                        &dxi_dy in &metrics.detj_dxi_dy(),
                        &hz in &y.hz())
            *a = dxi_dy * hz
        );
        op.diffxi(tmp.0.view(), tmp.1.view_mut());

        ndarray::azip!((b in &mut tmp.2,
                        &deta_dy in &metrics.detj_deta_dy(),
                        &hz in &y.hz())
            *b = deta_dy * hz
        );
        op.diffeta(tmp.2.view(), tmp.3.view_mut());

        ndarray::azip!((flux in &mut k.ex_mut(), &ax in &tmp.1, &by in &tmp.3)
            *flux = ax + by
        );
    }

    {
        // hz = -ey_x + ex_y
        ndarray::azip!((a in &mut tmp.0,
                        &dxi_dx in &metrics.detj_dxi_dx(),
                        &dxi_dy in &metrics.detj_dxi_dy(),
                        &ex in &y.ex(),
                        &ey in &y.ey())
            *a = dxi_dx * -ey + dxi_dy * ex
        );
        op.diffxi(tmp.0.view(), tmp.1.view_mut());

        ndarray::azip!((b in &mut tmp.2,
                        &deta_dx in &metrics.detj_deta_dx(),
                        &deta_dy in &metrics.detj_deta_dy(),
                        &ex in &y.ex(),
                        &ey in &y.ey())
            *b = deta_dx * -ey + deta_dy * ex
        );
        op.diffeta(tmp.2.view(), tmp.3.view_mut());

        ndarray::azip!((flux in &mut k.hz_mut(), &ax in &tmp.1, &by in &tmp.3)
            *flux = ax + by
        );
    }

    // ey = -hz_x
    {
        ndarray::azip!((a in &mut tmp.0,
                        &dxi_dx in &metrics.detj_dxi_dx(),
                        &hz in &y.hz())
            *a = dxi_dx * -hz
        );
        op.diffxi(tmp.0.view(), tmp.1.view_mut());

        azip!((b in &mut tmp.2,
                        &deta_dx in &metrics.detj_deta_dx(),
                        &hz in &y.hz())
            *b = deta_dx * -hz
        );
        op.diffeta(tmp.2.view(), tmp.3.view_mut());

        azip!((flux in &mut k.ey_mut(), &ax in &tmp.1, &by in &tmp.3)
            *flux = ax + by
        );
    }
}

fn dissipation<UO: UpwindOperator2d>(
    op: &UO,
    k: &mut Field,
    y: &Field,
    metrics: &Metrics,
    tmp: &mut (Array2<Float>, Array2<Float>, Array2<Float>, Array2<Float>),
) {
    // ex component
    {
        ndarray::azip!((a in &mut tmp.0,
                        &kx in &metrics.detj_dxi_dx(),
                        &ky in &metrics.detj_dxi_dy(),
                        &ex in &y.ex(),
                        &ey in &y.ey()) {
            let r = Float::hypot(kx, ky);
            *a = ky*ky/r * ex + -kx*ky/r*ey;
        });
        op.dissxi(tmp.0.view(), tmp.1.view_mut());

        ndarray::azip!((b in &mut tmp.2,
                    &kx in &metrics.detj_deta_dx(),
                    &ky in &metrics.detj_deta_dy(),
                    &ex in &y.ex(),
                    &ey in &y.ey()) {
            let r = Float::hypot(kx, ky);
            *b = ky*ky/r * ex + -kx*ky/r*ey;
        });
        op.disseta(tmp.2.view(), tmp.3.view_mut());

        ndarray::azip!((flux in &mut k.ex_mut(), &ax in &tmp.1, &by in &tmp.3)
            *flux += ax + by
        );
    }

    // hz component
    {
        ndarray::azip!((a in &mut tmp.0,
                        &kx in &metrics.detj_dxi_dx(),
                        &ky in &metrics.detj_dxi_dy(),
                        &hz in &y.hz()) {
            let r = Float::hypot(kx, ky);
            *a = r * hz;
        });
        op.dissxi(tmp.0.view(), tmp.1.view_mut());

        ndarray::azip!((b in &mut tmp.2,
                        &kx in &metrics.detj_deta_dx(),
                        &ky in &metrics.detj_deta_dy(),
                        &hz in &y.hz()) {
            let r = Float::hypot(kx, ky);
            *b = r * hz;
        });
        op.disseta(tmp.2.view(), tmp.3.view_mut());

        ndarray::azip!((flux in &mut k.hz_mut(), &ax in &tmp.1, &by in &tmp.3)
            *flux += ax + by
        );
    }

    // ey
    {
        ndarray::azip!((a in &mut tmp.0,
                        &kx in &metrics.detj_dxi_dx(),
                        &ky in &metrics.detj_dxi_dy(),
                        &ex in &y.ex(),
                        &ey in &y.ey()) {
            let r = Float::hypot(kx, ky);
            *a = -kx*ky/r * ex + kx*kx/r*ey;
        });
        op.dissxi(tmp.0.view(), tmp.1.view_mut());

        ndarray::azip!((b in &mut tmp.2,
                    &kx in &metrics.detj_deta_dx(),
                    &ky in &metrics.detj_deta_dy(),
                    &ex in &y.ex(),
                    &ey in &y.ey()) {
            let r = Float::hypot(kx, ky);
            *b = -kx*ky/r * ex + kx*kx/r*ey;
        });
        op.disseta(tmp.2.view(), tmp.3.view_mut());

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
fn SAT_characteristics<SBP: SbpOperator2d>(
    op: &SBP,
    k: &mut Field,
    y: &Field,
    metrics: &Metrics,
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
        let hinv = if op.is_h2xi() {
            (nx - 2) as Float / op.hxi()[0]
        } else {
            (nx - 1) as Float / op.hxi()[0]
        };
        for ((((mut k, v), g), &kx), &ky) in k
            .slice_mut(s![.., .., nx - 1])
            .gencolumns_mut()
            .into_iter()
            .zip(y.slice(s![.., .., nx - 1]).gencolumns())
            .zip(g.gencolumns())
            .zip(metrics.detj_dxi_dx().slice(s![.., nx - 1]))
            .zip(metrics.detj_dxi_dy().slice(s![.., nx - 1]))
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
        let hinv = if op.is_h2xi() {
            (nx - 2) as Float / op.hxi()[0]
        } else {
            (nx - 1) as Float / op.hxi()[0]
        };
        for ((((mut k, v), g), &kx), &ky) in k
            .slice_mut(s![.., .., 0])
            .gencolumns_mut()
            .into_iter()
            .zip(y.slice(s![.., .., 0]).gencolumns())
            .zip(g.gencolumns())
            .zip(metrics.detj_dxi_dx().slice(s![.., 0]))
            .zip(metrics.detj_dxi_dy().slice(s![.., 0]))
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
        let hinv = if op.is_h2eta() {
            (ny - 2) as Float / op.heta()[0]
        } else {
            (ny - 1) as Float / op.heta()[0]
        };
        for ((((mut k, v), g), &kx), &ky) in k
            .slice_mut(s![.., ny - 1, ..])
            .gencolumns_mut()
            .into_iter()
            .zip(y.slice(s![.., ny - 1, ..]).gencolumns())
            .zip(g.gencolumns())
            .zip(metrics.detj_deta_dx().slice(s![ny - 1, ..]))
            .zip(metrics.detj_deta_dy().slice(s![ny - 1, ..]))
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
        let hinv = if op.is_h2eta() {
            (ny - 2) as Float / op.heta()[0]
        } else {
            (ny - 1) as Float / op.heta()[0]
        };
        for ((((mut k, v), g), &kx), &ky) in k
            .slice_mut(s![.., 0, ..])
            .gencolumns_mut()
            .into_iter()
            .zip(y.slice(s![.., 0, ..]).gencolumns())
            .zip(g.gencolumns())
            .zip(metrics.detj_deta_dx().slice(s![0, ..]))
            .zip(metrics.detj_deta_dy().slice(s![0, ..]))
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
