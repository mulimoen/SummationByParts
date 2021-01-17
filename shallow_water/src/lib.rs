use ndarray::prelude::*;
use sbp::*;

mod characteristics;
use characteristics::{Aminus, Aplus, Bminus, Bplus};

const G: Float = 1.0;

#[derive(Clone, Debug)]
pub struct Field(Array3<Float>);

impl<'a> Into<ArrayView3<'a, Float>> for &'a Field {
    fn into(self) -> ArrayView3<'a, Float> {
        self.0.view()
    }
}

impl<'a> Into<ArrayViewMut3<'a, Float>> for &'a mut Field {
    fn into(self) -> ArrayViewMut3<'a, Float> {
        self.0.view_mut()
    }
}

impl Field {
    fn new(ny: usize, nx: usize) -> Self {
        Self(Array3::zeros((3, ny, nx)))
    }
    fn eta(&self) -> ArrayView2<Float> {
        self.0.slice(s![0, .., ..])
    }
    fn etau(&self) -> ArrayView2<Float> {
        self.0.slice(s![1, .., ..])
    }
    fn etav(&self) -> ArrayView2<Float> {
        self.0.slice(s![2, .., ..])
    }
    /*
    fn eta_mut(&mut self) -> ArrayViewMut2<Float> {
        self.0.slice_mut(s![0, .., ..])
    }
    fn etau_mut(&mut self) -> ArrayViewMut2<Float> {
        self.0.slice_mut(s![1, .., ..])
    }
    fn etav_mut(&mut self) -> ArrayViewMut2<Float> {
        self.0.slice_mut(s![2, .., ..])
    }
    */
    fn components_mut(
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

pub struct System {
    fnow: Field,
    fnext: Field,
    x: (Float, Float, usize),
    y: (Float, Float, usize),
    op: Box<dyn operators::SbpOperator2d>,
    k: [Field; 4],
}

impl System {
    pub fn new(x: (Float, Float, usize), y: (Float, Float, usize)) -> Self {
        let field = Field::new(y.2, x.2);
        Self {
            fnow: field.clone(),
            fnext: field.clone(),
            x,
            y,
            op: Box::new(operators::Upwind9),
            k: [field.clone(), field.clone(), field.clone(), field],
        }
    }
    pub fn nx(&self) -> usize {
        self.x.2
    }
    pub fn ny(&self) -> usize {
        self.y.2
    }
    pub fn eta(&self) -> ArrayView2<Float> {
        self.fnow.eta()
    }
    pub fn etau(&self) -> ArrayView2<Float> {
        self.fnow.etau()
    }
    pub fn etav(&self) -> ArrayView2<Float> {
        self.fnow.etav()
    }
    pub fn components_mut(
        &mut self,
    ) -> (
        ArrayViewMut2<Float>,
        ArrayViewMut2<Float>,
        ArrayViewMut2<Float>,
    ) {
        self.fnow.components_mut()
    }

    fn max_dt(&self) -> Float {
        0.1 * ((self.x.1 - self.x.0) / self.x.2 as Float)
            .min((self.y.1 - self.y.0) / self.y.2 as Float)
    }

    pub fn advance(&mut self) {
        let max_dt = self.max_dt();
        let op = &self.op;
        let rhs = move |next: &mut Field, now: &Field, _t: Float| {
            let (mut next_eta, mut next_etau, mut next_etav) = next.components_mut();
            next_eta.fill(0.0);
            next_etau.fill(0.0);
            next_etav.fill(0.0);

            let nx = next_eta.shape()[1];
            let ny = next_eta.shape()[0];

            if false {
                let eta = now.eta();
                for j in 0..ny {
                    for i in 0..nx {
                        if eta[(j, i)] <= 0.0 {
                            panic!("{} {}", j, i);
                        }
                    }
                }
            }

            let mut temp = Array2::<Float>::zeros((ny, nx));

            // E flux
            let mut temp_dx = temp.clone();
            azip!((dest in &mut temp, etau in now.etau()) {
                *dest = *etau;
            });
            op.diffxi(temp.view(), temp_dx.view_mut());
            next_eta.scaled_add(-1.0, &temp_dx);

            azip!((dest in &mut temp, eta in now.eta(), etau in now.etau()) {
                *dest = etau.powi(2)/eta + G*eta.powi(2)/2.0
            });
            op.diffxi(temp.view(), temp_dx.view_mut());
            next_etau.scaled_add(-1.0, &temp_dx);

            azip!((dest in &mut temp, eta in now.eta(), etau in now.etau(), etav in now.etav()) {
                *dest = etau*etav/eta
            });
            op.diffxi(temp.view(), temp_dx.view_mut());
            next_etav.scaled_add(-1.0, &temp_dx);

            // F flux
            let mut temp_dy = temp_dx;
            azip!((dest in &mut temp, etav in now.etav()) {
                *dest = *etav;
            });
            op.diffeta(temp.view(), temp_dy.view_mut());
            next_eta.scaled_add(-1.0, &temp_dy);

            azip!((dest in &mut temp, eta in now.eta(), etau in now.etau(), etav in now.etav()) {
                *dest = etau*etav/eta;
            });
            op.diffeta(temp.view(), temp_dy.view_mut());
            next_etau.scaled_add(-1.0, &temp_dy);

            azip!((dest in &mut temp, eta in now.eta(), etav in now.etav()) {
                *dest = etav.powi(2)/eta + G*eta.powi(2)/2.0
            });
            op.diffeta(temp.view(), temp_dy.view_mut());
            next_etav.scaled_add(-1.0, &temp_dy);

            // Upwind dissipation
            if false {
                if let Some(op) = op.upwind() {
                    let mut temp_dx = temp_dy;
                    azip!((dest in &mut temp, eta in now.eta(), etau in now.etau()) {
                        *dest = -(eta.powf(3.0/2.0)*G.sqrt() + etau.abs())/eta
                    });
                    op.dissxi(temp.view(), temp_dx.view_mut());
                    azip!((dest in &mut next_eta, eta in now.eta(), diss in &temp_dx) {
                        *dest -= eta*diss;
                    });
                    azip!((dest in &mut next_etau, etau in now.etau(), diss in &temp_dx) {
                        *dest -= etau*diss;
                    });
                    azip!((dest in &mut next_etav, etav in now.etav(), diss in &temp_dx) {
                        *dest -= etav*diss;
                    });

                    let mut temp_dy = temp_dx;
                    azip!((dest in &mut temp, eta in now.eta(), etav in now.etav()) {
                        *dest = -(eta.powf(3.0/2.0)*G.sqrt() + etav.abs())/eta
                    });
                    op.disseta(temp.view(), temp_dy.view_mut());
                    azip!((dest in &mut next_eta, eta in now.eta(), diss in &temp_dy) {
                        *dest -= eta*diss;
                    });
                    azip!((dest in &mut next_etau, etau in now.etau(), diss in &temp_dy) {
                        *dest -= etau*diss;
                    });
                    azip!((dest in &mut next_etav, etav in now.etav(), diss in &temp_dy) {
                        *dest -= etav*diss;
                    });
                }
            }

            // SAT boundaries
            #[derive(Debug)]
            enum Direction {
                North,
                South,
                East,
                West,
            }
            for dir in &[
                Direction::North,
                Direction::South,
                Direction::East,
                Direction::West,
            ] {
                let mut dest;
                let this;
                let other;
                match dir {
                    Direction::North => {
                        dest = next.0.slice_mut(s![.., ny - 1, ..]);
                        this = now.0.slice(s![.., ny - 1, ..]);
                        other = now.0.slice(s![.., 0_usize, ..]);
                    }
                    Direction::South => {
                        dest = next.0.slice_mut(s![.., 0_usize, ..]);
                        this = now.0.slice(s![.., 0_usize, ..]);
                        other = now.0.slice(s![.., ny - 1, ..]);
                    }
                    Direction::East => {
                        dest = next.0.slice_mut(s![.., .., nx - 1]);
                        this = now.0.slice(s![.., .., nx - 1]);
                        other = now.0.slice(s![.., .., 0_usize]);
                    }
                    Direction::West => {
                        dest = next.0.slice_mut(s![.., .., 0_usize]);
                        this = now.0.slice(s![.., .., 0_usize]);
                        other = now.0.slice(s![.., .., nx - 1]);
                    }
                }
                for ((mut dest, this), other) in dest
                    .axis_iter_mut(Axis(1))
                    .zip(this.axis_iter(Axis(1)))
                    .zip(other.axis_iter(Axis(1)))
                {
                    let tau = match dir {
                        Direction::North => 1.0,
                        Direction::South => -1.0,
                        Direction::East => 1.0,
                        Direction::West => -1.0,
                    };
                    let hinv = match dir {
                        Direction::North | Direction::South => {
                            if op.is_h2eta() {
                                (ny - 2) as Float / op.heta()[0]
                            } else {
                                (ny - 1) as Float / op.heta()[0]
                            }
                        }
                        Direction::East | Direction::West => {
                            if op.is_h2xi() {
                                (nx - 2) as Float / op.hxi()[0]
                            } else {
                                (nx - 1) as Float / op.hxi()[0]
                            }
                        }
                    };

                    let v = (this[0], this[1], this[2]);
                    let g = (other[0], other[1], other[2]);
                    let mat = match dir {
                        Direction::West => Aplus(v.0, v.1, v.2, G),
                        Direction::East => Aminus(v.0, v.1, v.2, G),
                        Direction::North => Bminus(v.0, v.1, v.2, G),
                        Direction::South => Bplus(v.0, v.1, v.2, G),
                    };

                    let q = [v.0 - g.0, v.1 - g.1, v.2 - g.2];

                    let mut res = [0.0; 3];
                    for j in 0..3 {
                        #[allow(clippy::needless_range_loop)]
                        for i in 0..3 {
                            res[j] += mat[j][i] * q[i];
                        }
                    }

                    for i in 0..3 {
                        dest[i] += tau * hinv * res[i];
                    }
                }
            }
            log::trace!("Iteration complete");
        };
        integrate::integrate::<integrate::Rk4, _, _, _>(
            rhs,
            &self.fnow,
            &mut self.fnext,
            &mut 0.0,
            max_dt,
            &mut self.k[..],
        );

        std::mem::swap(&mut self.fnow, &mut self.fnext)
    }
}

#[test]
fn test_advance() {
    let mut sys = System::new((0.0, 1.0, 50), (0.0, 1.0, 50));
    sys.fnow.components_mut().0.fill(1.0);
    for _ in 0..10 {
        sys.advance();
    }
    println!("{:?}", sys.fnow);
}
