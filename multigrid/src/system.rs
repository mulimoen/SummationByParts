use crate::parsing;
use crate::utils::Direction;
use core::ops::Deref;
use crossbeam_channel::{Receiver, Select, Sender};
use euler::{
    eval::{self, Evaluator},
    Diff, Field, VortexParameters, WorkBuffers,
};
use ndarray::Array2;
use sbp::grid::{Grid, Metrics};
use sbp::operators::{InterpolationOperator, SbpOperator2d};
use sbp::*;

pub struct BaseSystem {
    pub names: Vec<String>,
    pub grids: Vec<grid::Grid>,
    pub time: Float,
    pub boundary_conditions: Vec<euler::BoundaryCharacteristics>,
    pub initial_conditions: crate::parsing::InitialConditions,
    pub operators: Vec<Box<dyn SbpOperator2d>>,
    pub output: hdf5::File,
}

impl BaseSystem {
    pub fn new(
        names: Vec<String>,
        grids: Vec<grid::Grid>,
        time: Float,
        operators: Vec<Box<dyn SbpOperator2d>>,
        boundary_conditions: Vec<euler::BoundaryCharacteristics>,
        initial_conditions: crate::parsing::InitialConditions,
        output: std::path::PathBuf,
    ) -> Self {
        let output = hdf5::File::create(output).unwrap();
        output
            .new_dataset::<u64>()
            .resizable(true)
            .chunk((1,))
            .create("t", (0,))
            .unwrap();
        Self {
            names,
            grids,
            time,
            boundary_conditions,
            initial_conditions,
            operators,
            output,
        }
    }
    #[allow(clippy::many_single_char_names)]
    pub fn create(self) -> System {
        let fnow = self
            .grids
            .iter()
            .map(|g| euler::Field::new(g.ny(), g.nx()))
            .collect::<Vec<_>>();
        let fnext = fnow.clone();
        let wb = self
            .grids
            .iter()
            .map(|g| euler::WorkBuffers::new(g.ny(), g.nx()))
            .collect();
        let k = self
            .grids
            .iter()
            .map(|g| euler::Diff::zeros((g.ny(), g.nx())))
            .collect::<Vec<_>>();
        let k = [k.clone(), k.clone(), k.clone(), k];
        let metrics = self
            .grids
            .iter()
            .zip(&self.operators)
            .map(|(g, op)| g.metrics(&**op).unwrap())
            .collect::<Vec<_>>();

        let eb = self
            .boundary_conditions
            .iter()
            .zip(&self.grids)
            .map(|(bt, grid)| euler::BoundaryStorage::new(bt, grid))
            .collect();

        let mut outputs = Vec::with_capacity(self.grids.len());
        for (name, grid) in self.names.iter().zip(&self.grids) {
            let g = self.output.create_group(name).unwrap();
            g.link_soft("/t", "t").unwrap();

            let add_dim = |name| {
                g.new_dataset::<Float>()
                    .chunk((grid.ny(), grid.nx()))
                    .gzip(9)
                    .create(name, (grid.ny(), grid.nx()))
            };
            let xds = add_dim("x").unwrap();
            xds.write(grid.x()).unwrap();
            let yds = add_dim("y").unwrap();
            yds.write(grid.y()).unwrap();

            let add_var = |name| {
                g.new_dataset::<Float>()
                    .gzip(3)
                    .shuffle(true)
                    .chunk((1, grid.ny(), grid.nx()))
                    .resizable(true)
                    .create(name, (0, grid.ny(), grid.nx()))
            };
            add_var("rho").unwrap();
            add_var("rhou").unwrap();
            add_var("rhov").unwrap();
            add_var("e").unwrap();
            outputs.push(g);
        }

        let mut sys = SingleThreadedSystem {
            fnow,
            fnext,
            k,
            wb,
            grids: self.grids,
            metrics,
            bt: self.boundary_conditions,
            eb,
            time: self.time,
            dt: Float::NAN,
            operators: self.operators,
            output: (self.output, outputs),
        };
        match &self.initial_conditions {
            /*
            parsing::InitialConditions::File(f) => {
                for grid in &sys.grids {
                    // Copy initial conditions from file, requires name of field
                    todo!()
                }
            }
            */
            parsing::InitialConditions::Vortex(vortexparams) => sys.vortex(0.0, vortexparams),
            parsing::InitialConditions::Expressions(expr) => {
                let t = 0.0;
                for (grid, field) in sys.grids.iter().zip(sys.fnow.iter_mut()) {
                    // Evaluate the expressions on all variables
                    let x = grid.x();
                    let y = grid.y();
                    let (rho, rhou, rhov, e) = field.components_mut();
                    (*expr).evaluate(t, x, y, rho, rhou, rhov, e);
                }
            }
        }
        System::SingleThreaded(sys)
    }

    /// Spreads the computation over n threads, in a thread per grid way.
    pub fn create_distributed(self) -> System {
        let nthreads = self.grids.len();

        // Build up the boundary conditions
        let mut push_channels: Vec<Direction<Option<Sender<Array2<Float>>>>> =
            Vec::with_capacity(nthreads);
        let mut pull_channels: Vec<Direction<Option<Receiver<Array2<Float>>>>> =
            vec![Direction::default(); nthreads];

        // Build the set of communicators between boundaries
        for wb in &self.boundary_conditions {
            let mut local_push = Direction::default();
            if let euler::BoundaryCharacteristic::Grid(i)
            | euler::BoundaryCharacteristic::Interpolate(i, _) = wb.north()
            {
                let (s, r) = crossbeam_channel::bounded(1);
                pull_channels[*i]
                    .south_mut()
                    .replace(r)
                    .and_then::<(), _>(|_| panic!("channel is already present"));
                *local_push.north_mut() = Some(s);
            }
            if let euler::BoundaryCharacteristic::Grid(i)
            | euler::BoundaryCharacteristic::Interpolate(i, _) = wb.south()
            {
                let (s, r) = crossbeam_channel::bounded(1);
                pull_channels[*i]
                    .north_mut()
                    .replace(r)
                    .and_then::<(), _>(|_| panic!("channel is already present"));
                *local_push.south_mut() = Some(s);
            }
            if let euler::BoundaryCharacteristic::Grid(i)
            | euler::BoundaryCharacteristic::Interpolate(i, _) = wb.east()
            {
                let (s, r) = crossbeam_channel::bounded(1);
                pull_channels[*i]
                    .west_mut()
                    .replace(r)
                    .and_then::<(), _>(|_| panic!("channel is already present"));
                *local_push.east_mut() = Some(s);
            }
            if let euler::BoundaryCharacteristic::Grid(i)
            | euler::BoundaryCharacteristic::Interpolate(i, _) = wb.west()
            {
                let (s, r) = crossbeam_channel::bounded(1);
                pull_channels[*i]
                    .east_mut()
                    .replace(r)
                    .and_then::<(), _>(|_| panic!("channel is already present"));
                *local_push.west_mut() = Some(s);
            }

            push_channels.push(local_push);
        }

        let (master_send, master_recv) = crossbeam_channel::unbounded();

        let mut tids = Vec::with_capacity(nthreads);
        let mut communicators = Vec::with_capacity(nthreads);

        for (id, (((((name, grid), sbp), bt), chan), push)) in self
            .names
            .into_iter()
            .zip(self.grids.into_iter())
            .zip(self.operators.into_iter())
            .zip(self.boundary_conditions)
            .zip(pull_channels)
            .zip(push_channels)
            .enumerate()
        {
            let builder = std::thread::Builder::new().name(format!("eulersolver: {}", name));

            let boundary_conditions = bt.zip(chan).map(|(bt, chan)| match bt {
                euler::BoundaryCharacteristic::This => DistributedBoundaryConditions::This,
                euler::BoundaryCharacteristic::Grid(_) => {
                    DistributedBoundaryConditions::Channel(chan.unwrap())
                }
                euler::BoundaryCharacteristic::Interpolate(_, int_op) => {
                    DistributedBoundaryConditions::Interpolate(chan.unwrap(), int_op)
                }
                euler::BoundaryCharacteristic::MultiGrid(_) => unimplemented!(),
                euler::BoundaryCharacteristic::Vortex(vp) => {
                    DistributedBoundaryConditions::Vortex(vp)
                }
                euler::BoundaryCharacteristic::Eval(eval) => {
                    DistributedBoundaryConditions::Eval(eval)
                }
            });

            let master_send = master_send.clone();

            let (t_send, t_recv) = crossbeam_channel::unbounded();
            communicators.push(t_send);
            let time = self.time;

            let output = self.output.clone();

            tids.push(
                builder
                    .spawn(move || {
                        let (ny, nx) = (grid.ny(), grid.nx());
                        let current = Field::new(ny, nx);
                        let fut = current.clone();
                        let k = [
                            Diff::zeros((ny, nx)),
                            Diff::zeros((ny, nx)),
                            Diff::zeros((ny, nx)),
                            Diff::zeros((ny, nx)),
                        ];
                        let metrics = grid.metrics(sbp.deref()).unwrap();

                        let wb = WorkBuffers::new(ny, nx);

                        let g = {
                            let g = output.create_group(&name).unwrap();
                            g.link_soft("/t", "t").unwrap();
                            let add_dim = |name| {
                                g.new_dataset::<Float>()
                                    .chunk((grid.ny(), grid.nx()))
                                    .gzip(9)
                                    .create(name, (grid.ny(), grid.nx()))
                            };
                            let xds = add_dim("x").unwrap();
                            xds.write(grid.x()).unwrap();
                            let yds = add_dim("y").unwrap();
                            yds.write(grid.y()).unwrap();

                            let add_var = |name| {
                                g.new_dataset::<Float>()
                                    .gzip(3)
                                    .shuffle(true)
                                    .chunk((1, grid.ny(), grid.nx()))
                                    .resizable(true)
                                    .create(name, (0, grid.ny(), grid.nx()))
                            };
                            add_var("rho").unwrap();
                            add_var("rhou").unwrap();
                            add_var("rhov").unwrap();
                            add_var("e").unwrap();
                            g
                        };

                        let mut sys = DistributedSystemPart {
                            current,
                            fut,
                            k,
                            boundary_conditions,
                            grid: (grid, metrics),
                            output: g,
                            push,
                            sbp,
                            t: time,
                            dt: Float::NAN,

                            _name: name,
                            id,

                            recv: t_recv,
                            send: master_send,

                            wb,
                            wb_ns: Array2::zeros((4, nx)),
                            wb_ew: Array2::zeros((4, ny)),
                        };

                        // init and send maxdt
                        // receive maxdt
                        sys.run();
                    })
                    .unwrap(),
            );
        }

        System::MultiThreaded(DistributedSystem {
            sys: tids,
            recv: master_recv,
            send: communicators,
            output: self.output,
        })
    }
}

pub enum System {
    SingleThreaded(SingleThreadedSystem),
    MultiThreaded(DistributedSystem),
}

impl System {
    pub fn max_dt(&self) -> Float {
        match self {
            Self::SingleThreaded(sys) => sys.max_dt(),
            Self::MultiThreaded(sys) => {
                for send in &sys.send {
                    send.send(MsgFromHost::DtRequest).unwrap();
                }
                let mut max_dt = Float::MAX;
                let mut to_receive = sys.sys.len();
                while to_receive != 0 {
                    let dt = match sys.recv.recv().unwrap() {
                        (_, MsgToHost::MaxDt(dt)) => dt,
                        _ => unreachable!(),
                    };
                    max_dt = max_dt.min(dt);
                    to_receive -= 1;
                }
                max_dt
            }
        }
    }
    pub fn set_dt(&mut self, dt: Float) {
        match self {
            Self::SingleThreaded(sys) => sys.dt = dt,
            Self::MultiThreaded(sys) => {
                for tid in &sys.send {
                    tid.send(MsgFromHost::DtSet(dt)).unwrap()
                }
            }
        }
    }
    pub fn advance(&mut self, nsteps: u64) {
        match self {
            Self::SingleThreaded(sys) => sys.advance(nsteps),
            Self::MultiThreaded(sys) => sys.advance(nsteps),
        }
    }
    pub fn output(&self, ntime: u64) {
        match self {
            Self::SingleThreaded(sys) => sys.output(ntime),
            Self::MultiThreaded(sys) => sys.output(ntime),
        }
    }
}

pub struct SingleThreadedSystem {
    pub fnow: Vec<euler::Field>,
    pub fnext: Vec<euler::Field>,
    pub wb: Vec<euler::WorkBuffers>,
    pub k: [Vec<euler::Diff>; 4],
    pub grids: Vec<grid::Grid>,
    pub metrics: Vec<grid::Metrics>,
    pub bt: Vec<euler::BoundaryCharacteristics>,
    pub eb: Vec<euler::BoundaryStorage>,
    pub time: Float,
    pub dt: Float,
    pub operators: Vec<Box<dyn SbpOperator2d>>,
    pub output: (hdf5::File, Vec<hdf5::Group>),
}

impl integrate::Integrable for SingleThreadedSystem {
    type State = Vec<euler::Field>;
    type Diff = Vec<euler::Diff>;

    fn scaled_add(s: &mut Self::State, o: &Self::Diff, scale: Float) {
        s.iter_mut()
            .zip(o.iter())
            .for_each(|(s, o)| euler::Field::scaled_add(s, o, scale))
    }
}

impl SingleThreadedSystem {
    pub fn vortex(&mut self, t: Float, vortex_params: &euler::VortexParameters) {
        for (f, g) in self.fnow.iter_mut().zip(&self.grids) {
            f.vortex(g.x(), g.y(), t, vortex_params);
        }
    }

    pub fn advance(&mut self, nsteps: u64) {
        for _ in 0..nsteps {
            self.advance_single_step(self.dt)
        }
    }

    pub fn advance_single_step(&mut self, dt: Float) {
        let metrics = &self.metrics;
        let grids = &self.grids;
        let bt = &self.bt;
        let wb = &mut self.wb;
        let eb = &mut self.eb;
        let operators = &self.operators;

        let rhs = move |fut: &mut Vec<euler::Diff>, prev: &Vec<euler::Field>, time: Float| {
            let prev_all = &prev;
            for (((((((fut, prev), wb), grid), metrics), op), bt), eb) in fut
                .iter_mut()
                .zip(prev.iter())
                .zip(wb.iter_mut())
                .zip(grids)
                .zip(metrics.iter())
                .zip(operators.iter())
                .zip(bt.iter())
                .zip(eb.iter_mut())
            {
                let bc = euler::boundary_extracts(prev_all, bt, prev, grid, eb, time);
                if op.upwind().is_some() {
                    euler::RHS_upwind(&**op, fut, prev, metrics, &bc, &mut wb.0);
                } else {
                    euler::RHS_trad(&**op, fut, prev, metrics, &bc, &mut wb.0);
                }
            }
        };

        integrate::integrate::<integrate::Rk4, SingleThreadedSystem, _>(
            rhs,
            &self.fnow,
            &mut self.fnext,
            &mut self.time,
            dt,
            &mut self.k,
        );

        std::mem::swap(&mut self.fnow, &mut self.fnext);
    }

    /// Suggested maximum dt for this problem
    pub fn max_dt(&self) -> Float {
        let is_h2 = self
            .operators
            .iter()
            .any(|op| op.is_h2xi() || op.is_h2eta());
        let c_max = if is_h2 { 0.5 } else { 1.0 };
        let mut max_dt: Float = Float::INFINITY;

        for (field, metrics) in self.fnow.iter().zip(self.metrics.iter()) {
            let nx = field.nx();
            let ny = field.ny();

            let rho = field.rho();
            let rhou = field.rhou();
            let rhov = field.rhov();

            let mut max_u: Float = 0.0;
            let mut max_v: Float = 0.0;

            for ((((((rho, rhou), rhov), detj_dxi_dx), detj_dxi_dy), detj_deta_dx), detj_deta_dy) in
                rho.iter()
                    .zip(rhou.iter())
                    .zip(rhov.iter())
                    .zip(metrics.detj_dxi_dx())
                    .zip(metrics.detj_dxi_dy())
                    .zip(metrics.detj_deta_dx())
                    .zip(metrics.detj_deta_dy())
            {
                let u = rhou / rho;
                let v = rhov / rho;

                let uhat: Float = detj_dxi_dx * u + detj_dxi_dy * v;
                let vhat: Float = detj_deta_dx * u + detj_deta_dy * v;

                max_u = max_u.max(uhat.abs());
                max_v = max_v.max(vhat.abs());
            }

            let dx = 1.0 / nx as Float;
            let dy = 1.0 / ny as Float;

            let c_dt = Float::max(max_u / dx, max_v / dy);

            max_dt = Float::min(max_dt, c_max / c_dt);
        }

        max_dt
    }

    pub fn output(&self, ntime: u64) {
        let tds = self.output.0.dataset("t").unwrap();
        let tpos = tds.size();
        tds.resize((tpos + 1,)).unwrap();
        tds.write_slice(&[ntime], ndarray::s![tpos..tpos + 1])
            .unwrap();
        for (group, fnow) in self.output.1.iter().zip(&self.fnow) {
            let (ny, nx) = (fnow.ny(), fnow.nx());
            let rhods = group.dataset("rho").unwrap();
            let rhouds = group.dataset("rhou").unwrap();
            let rhovds = group.dataset("rhov").unwrap();
            let eds = group.dataset("e").unwrap();

            let (rho, rhou, rhov, e) = fnow.components();
            rhods.resize((tpos + 1, ny, nx)).unwrap();
            rhods.write_slice(rho, ndarray::s![tpos, .., ..]).unwrap();

            rhouds.resize((tpos + 1, ny, nx)).unwrap();
            rhouds.write_slice(rhou, ndarray::s![tpos, .., ..]).unwrap();

            rhovds.resize((tpos + 1, ny, nx)).unwrap();
            rhovds.write_slice(rhov, ndarray::s![tpos, .., ..]).unwrap();

            eds.resize((tpos + 1, ny, nx)).unwrap();
            eds.write_slice(e, ndarray::s![tpos, .., ..]).unwrap();
        }
    }
}

pub struct DistributedSystem {
    recv: Receiver<(usize, MsgToHost)>,
    send: Vec<Sender<MsgFromHost>>,
    /// All threads should be joined to mark the end of the computation
    sys: Vec<std::thread::JoinHandle<()>>,
    output: hdf5::File,
}

impl DistributedSystem {
    pub fn advance(&mut self, ntime: u64) {
        for tid in &self.send {
            tid.send(MsgFromHost::Advance(ntime)).unwrap();
        }
    }
    pub fn output(&self, ntime: u64) {
        for tid in &self.send {
            tid.send(MsgFromHost::Output(ntime)).unwrap();
        }
        let tds = self.output.dataset("t").unwrap();
        let tpos = tds.size();
        tds.resize((tpos + 1,)).unwrap();
        tds.write_slice(&[ntime], ndarray::s![tpos..tpos + 1])
            .unwrap();
    }
}

impl Drop for DistributedSystem {
    fn drop(&mut self) {
        for tid in &self.send {
            tid.send(MsgFromHost::Stop).unwrap();
        }
        let handles = std::mem::take(&mut self.sys);
        for tid in handles {
            tid.join().unwrap()
        }
    }
}

enum MsgFromHost {
    Advance(u64),
    DtRequest,
    DtSet(Float),
    Output(u64),
    Stop,
}

enum MsgToHost {
    MaxDt(Float),
    CurrentTimestep(u64),
}

// #[derive(Debug)]
pub enum DistributedBoundaryConditions {
    This,

    Vortex(VortexParameters),
    Eval(std::sync::Arc<dyn eval::Evaluator<ndarray::Ix1>>),

    Interpolate(Receiver<Array2<Float>>, Box<dyn InterpolationOperator>),
    Channel(Receiver<Array2<Float>>),
}

type PushCommunicator = Option<Sender<Array2<Float>>>;

struct DistributedSystemPart {
    grid: (Grid, Metrics),
    sbp: Box<dyn SbpOperator2d + 'static>,

    boundary_conditions: Direction<DistributedBoundaryConditions>,
    /// Subscribers to the boundaries of self
    push: Direction<PushCommunicator>,

    current: Field,
    fut: Field,

    t: Float,
    dt: Float,

    _name: String,
    id: usize,
    recv: Receiver<MsgFromHost>,
    send: Sender<(usize, MsgToHost)>,

    output: hdf5::Group,

    k: [Diff; 4],
    wb: WorkBuffers,
    /// Work buffer for north/south boundary
    wb_ns: Array2<Float>,
    /// Work buffer for east/west boundary
    wb_ew: Array2<Float>,
}

impl DistributedSystemPart {
    fn run(&mut self) {
        loop {
            match self.recv.recv().unwrap() {
                MsgFromHost::DtSet(dt) => self.dt = dt,
                MsgFromHost::DtRequest => todo!(),
                MsgFromHost::Advance(ntime) => self.advance(ntime),
                MsgFromHost::Output(ntime) => self.output(ntime),
                MsgFromHost::Stop => return,
            }
        }
    }

    fn output(&mut self, _ntime: u64) {
        let (ny, nx) = (self.current.ny(), self.current.nx());
        let rhods = self.output.dataset("rho").unwrap();
        let rhouds = self.output.dataset("rhou").unwrap();
        let rhovds = self.output.dataset("rhov").unwrap();
        let eds = self.output.dataset("e").unwrap();

        let (rho, rhou, rhov, e) = self.current.components();
        let tpos = rhods.size() / (ny * nx);
        rhods.resize((tpos + 1, ny, nx)).unwrap();
        rhods.write_slice(rho, ndarray::s![tpos, .., ..]).unwrap();

        rhouds.resize((tpos + 1, ny, nx)).unwrap();
        rhouds.write_slice(rhou, ndarray::s![tpos, .., ..]).unwrap();

        rhovds.resize((tpos + 1, ny, nx)).unwrap();
        rhovds.write_slice(rhov, ndarray::s![tpos, .., ..]).unwrap();

        eds.resize((tpos + 1, ny, nx)).unwrap();
        eds.write_slice(e, ndarray::s![tpos, .., ..]).unwrap();
    }

    fn advance(&mut self, ntime: u64) {
        for ntime in 0..ntime {
            self.send
                .send((self.id, MsgToHost::CurrentTimestep(ntime)))
                .unwrap();
            let metrics = &self.grid.1;
            let wb = &mut self.wb.0;
            let sbp = &self.sbp;
            let push = &self.push;
            let boundary_conditions = &self.boundary_conditions;
            let grid = &self.grid.0;
            let wb_ns = &mut self.wb_ns;
            let wb_ew = &mut self.wb_ew;

            let rhs = |k: &mut euler::Diff, y: &euler::Field, time: Float| {
                // Send off the boundaries optimistically, in case some grid is ready
                if let Some(s) = &push.north {
                    s.send(y.north().to_owned()).unwrap()
                }
                if let Some(s) = &push.south {
                    s.send(y.south().to_owned()).unwrap()
                }
                if let Some(s) = &push.east {
                    s.send(y.east().to_owned()).unwrap()
                }
                if let Some(s) = &push.west {
                    s.send(y.west().to_owned()).unwrap()
                }

                // This computation does not depend on the boundaries
                euler::RHS_no_SAT(sbp.deref(), k, y, metrics, wb);

                // Get boundaries, but be careful and maximise the amount of work which can be
                // performed before we have all of them, whilst ensuring threads can sleep for as
                // long as possible
                let mut select = Select::new();
                let mut selectable = 0;
                let recv_north = match boundary_conditions.north() {
                    DistributedBoundaryConditions::Channel(r)
                    | DistributedBoundaryConditions::Interpolate(r, _) => {
                        selectable += 1;
                        Some(select.recv(r))
                    }
                    DistributedBoundaryConditions::This => {
                        euler::SAT_north(sbp.deref(), k, y, metrics, y.south());
                        None
                    }
                    DistributedBoundaryConditions::Vortex(vp) => {
                        let mut fiter = wb_ns.outer_iter_mut();
                        let (rho, rhou, rhov, e) = (
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                        );
                        let (gx, gy) = grid.north();
                        vp.evaluate(time, gx, gy, rho, rhou, rhov, e);

                        euler::SAT_north(sbp.deref(), k, y, metrics, wb_ns.view());
                        None
                    }
                    DistributedBoundaryConditions::Eval(eval) => {
                        let mut fiter = wb_ns.outer_iter_mut();
                        let (rho, rhou, rhov, e) = (
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                        );
                        let (gx, gy) = grid.north();
                        eval.evaluate(time, gx, gy, rho, rhou, rhov, e);
                        euler::SAT_north(sbp.deref(), k, y, metrics, wb_ns.view());
                        None
                    }
                };
                let recv_south = match boundary_conditions.south() {
                    DistributedBoundaryConditions::Channel(r)
                    | DistributedBoundaryConditions::Interpolate(r, _) => {
                        selectable += 1;
                        Some(select.recv(r))
                    }
                    DistributedBoundaryConditions::This => {
                        euler::SAT_south(sbp.deref(), k, y, metrics, y.north());
                        None
                    }
                    DistributedBoundaryConditions::Vortex(vp) => {
                        let mut fiter = wb_ns.outer_iter_mut();
                        let (rho, rhou, rhov, e) = (
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                        );
                        let (gx, gy) = grid.south();
                        vp.evaluate(time, gx, gy, rho, rhou, rhov, e);

                        euler::SAT_south(sbp.deref(), k, y, metrics, wb_ns.view());
                        None
                    }
                    DistributedBoundaryConditions::Eval(eval) => {
                        let mut fiter = wb_ns.outer_iter_mut();
                        let (rho, rhou, rhov, e) = (
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                        );
                        let (gx, gy) = grid.south();
                        eval.evaluate(time, gx, gy, rho, rhou, rhov, e);
                        euler::SAT_south(sbp.deref(), k, y, metrics, wb_ns.view());
                        None
                    }
                };
                let recv_east = match boundary_conditions.east() {
                    DistributedBoundaryConditions::Channel(r)
                    | DistributedBoundaryConditions::Interpolate(r, _) => {
                        selectable += 1;
                        Some(select.recv(r))
                    }
                    DistributedBoundaryConditions::This => {
                        euler::SAT_east(sbp.deref(), k, y, metrics, y.west());
                        None
                    }
                    DistributedBoundaryConditions::Vortex(vp) => {
                        let mut fiter = wb_ew.outer_iter_mut();
                        let (rho, rhou, rhov, e) = (
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                        );
                        let (gx, gy) = grid.east();
                        vp.evaluate(time, gx, gy, rho, rhou, rhov, e);

                        euler::SAT_east(sbp.deref(), k, y, metrics, wb_ew.view());
                        None
                    }
                    DistributedBoundaryConditions::Eval(eval) => {
                        let mut fiter = wb_ew.outer_iter_mut();
                        let (rho, rhou, rhov, e) = (
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                        );
                        let (gx, gy) = grid.east();
                        eval.evaluate(time, gx, gy, rho, rhou, rhov, e);
                        euler::SAT_east(sbp.deref(), k, y, metrics, wb_ew.view());
                        None
                    }
                };
                let recv_west = match boundary_conditions.west() {
                    DistributedBoundaryConditions::Channel(r)
                    | DistributedBoundaryConditions::Interpolate(r, _) => {
                        selectable += 1;
                        Some(select.recv(r))
                    }
                    DistributedBoundaryConditions::This => {
                        euler::SAT_west(sbp.deref(), k, y, metrics, y.east());
                        None
                    }
                    DistributedBoundaryConditions::Vortex(vp) => {
                        let mut fiter = wb_ew.outer_iter_mut();
                        let (rho, rhou, rhov, e) = (
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                        );
                        let (gx, gy) = grid.west();
                        vp.evaluate(time, gx, gy, rho, rhou, rhov, e);

                        euler::SAT_west(sbp.deref(), k, y, metrics, wb_ew.view());
                        None
                    }
                    DistributedBoundaryConditions::Eval(eval) => {
                        let mut fiter = wb_ew.outer_iter_mut();
                        let (rho, rhou, rhov, e) = (
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                        );
                        let (gx, gy) = grid.west();
                        eval.evaluate(time, gx, gy, rho, rhou, rhov, e);
                        euler::SAT_west(sbp.deref(), k, y, metrics, wb_ew.view());
                        None
                    }
                };

                // Get an item off each channel, waiting minimally before processing that boundary.
                // The waiting ensures other grids can be processed by the core in case of
                // oversubscription (in case of a more grids than core scenario)
                // This minimises the amount of time waiting on boundary conditions
                while selectable != 0 {
                    let s = select.select();
                    let sindex = s.index();
                    match Some(sindex) {
                        x if x == recv_north => match boundary_conditions.north() {
                            DistributedBoundaryConditions::Channel(r) => {
                                let r = s.recv(r).unwrap();
                                euler::SAT_north(sbp.deref(), k, y, metrics, r.view());
                            }
                            DistributedBoundaryConditions::Interpolate(r, int_op) => {
                                let r = s.recv(r).unwrap();
                                let is_fine2coarse = r.shape()[1] > wb_ns.shape()[1];
                                for (mut to, from) in wb_ns.outer_iter_mut().zip(r.outer_iter()) {
                                    if is_fine2coarse {
                                        int_op.fine2coarse(from.view(), to.view_mut());
                                    } else {
                                        int_op.coarse2fine(from.view(), to.view_mut());
                                    }
                                }
                                euler::SAT_north(sbp.deref(), k, y, metrics, wb_ns.view());
                            }
                            _ => unreachable!(),
                        },
                        x if x == recv_south => match boundary_conditions.south() {
                            DistributedBoundaryConditions::Channel(r) => {
                                let r = s.recv(r).unwrap();
                                euler::SAT_south(sbp.deref(), k, y, metrics, r.view());
                            }
                            DistributedBoundaryConditions::Interpolate(r, int_op) => {
                                let r = s.recv(r).unwrap();
                                let is_fine2coarse = r.shape()[1] > wb_ns.shape()[1];
                                for (mut to, from) in wb_ns.outer_iter_mut().zip(r.outer_iter()) {
                                    if is_fine2coarse {
                                        int_op.fine2coarse(from.view(), to.view_mut());
                                    } else {
                                        int_op.coarse2fine(from.view(), to.view_mut());
                                    }
                                }
                                euler::SAT_south(sbp.deref(), k, y, metrics, wb_ns.view());
                            }
                            _ => unreachable!(),
                        },
                        x if x == recv_west => match boundary_conditions.west() {
                            DistributedBoundaryConditions::Channel(r) => {
                                let r = s.recv(r).unwrap();
                                euler::SAT_west(sbp.deref(), k, y, metrics, r.view());
                            }
                            DistributedBoundaryConditions::Interpolate(r, int_op) => {
                                let r = s.recv(r).unwrap();
                                let is_fine2coarse = r.shape()[1] > wb_ew.shape()[1];
                                for (mut to, from) in wb_ew.outer_iter_mut().zip(r.outer_iter()) {
                                    if is_fine2coarse {
                                        int_op.fine2coarse(from.view(), to.view_mut());
                                    } else {
                                        int_op.coarse2fine(from.view(), to.view_mut());
                                    }
                                }
                                euler::SAT_west(sbp.deref(), k, y, metrics, wb_ew.view());
                            }
                            _ => unreachable!(),
                        },
                        x if x == recv_east => match boundary_conditions.east() {
                            DistributedBoundaryConditions::Channel(r) => {
                                let r = s.recv(r).unwrap();
                                euler::SAT_east(sbp.deref(), k, y, metrics, r.view());
                            }
                            DistributedBoundaryConditions::Interpolate(r, int_op) => {
                                let r = s.recv(r).unwrap();
                                let is_fine2coarse = r.shape()[1] > wb_ew.shape()[1];
                                for (mut to, from) in wb_ew.outer_iter_mut().zip(r.outer_iter()) {
                                    if is_fine2coarse {
                                        int_op.fine2coarse(from.view(), to.view_mut());
                                    } else {
                                        int_op.coarse2fine(from.view(), to.view_mut());
                                    }
                                }
                                euler::SAT_east(sbp.deref(), k, y, metrics, wb_ew.view());
                            }
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    }
                    select.remove(sindex);
                    selectable -= 1;
                }
            };
            integrate::integrate::<integrate::Rk4, Field, _>(
                rhs,
                &self.current,
                &mut self.fut,
                &mut self.t,
                self.dt,
                &mut self.k,
            )
        }
    }
}
