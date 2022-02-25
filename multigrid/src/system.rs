use crate::parsing;
use crate::utils::Direction;
use arrayvec::ArrayVec;
use core::ops::Deref;
use crossbeam_channel::{Receiver, Sender};
use euler::{
    eval::{self, Evaluator},
    Diff, Field, VortexParameters, WorkBuffers,
};
use ndarray::Array2;
use parking_lot::{Condvar, Mutex};
use sbp::grid::{Grid, Metrics};
use sbp::operators::{InterpolationOperator, SbpOperator2d};
use sbp::*;
use std::sync::Arc;

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
            .shape((0..,))
            .chunk((1,))
            .create("t")
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
                    .deflate(9)
                    .shape((grid.ny(), grid.nx()))
                    .create(name)
            };
            let xds = add_dim("x").unwrap();
            xds.write(grid.x()).unwrap();
            let yds = add_dim("y").unwrap();
            yds.write(grid.y()).unwrap();

            let add_var = |name| {
                g.new_dataset::<Float>()
                    .shuffle()
                    .deflate(3)
                    .chunk((1, grid.ny(), grid.nx()))
                    .shape((0.., grid.ny(), grid.nx()))
                    .create(name)
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
            progressbar: None,
            initial_conditions: self.initial_conditions.clone(),
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
        let mut pull = Vec::<Arc<Communicator>>::with_capacity(nthreads);
        for _ in 0..nthreads {
            pull.push(Arc::new(Communicator {
                cvar: Condvar::new(),
                data: Mutex::new(Direction::splat(()).map(|_| ArrayVec::new())),
            }));
        }

        // Build the set of communicators between boundaries
        let mut push = Vec::<Direction<Option<Arc<Communicator>>>>::with_capacity(nthreads);
        for wb in &self.boundary_conditions {
            let local_push = wb.as_ref().map(|bc| {
                if let euler::BoundaryCharacteristic::Grid(i)
                | euler::BoundaryCharacteristic::Interpolate(i, _) = bc
                {
                    Some(pull[*i].clone())
                } else {
                    None
                }
            });

            push.push(local_push);
        }

        let (master_send, master_recv) = crossbeam_channel::unbounded();

        let mut tids = Vec::with_capacity(nthreads);
        let mut communicators = Vec::with_capacity(nthreads);

        for (id, (((((name, grid), sbp), bt), pull), push)) in self
            .names
            .into_iter()
            .zip(self.grids.into_iter())
            .zip(self.operators.into_iter())
            .zip(self.boundary_conditions)
            .zip(pull)
            .zip(push)
            .enumerate()
        {
            let builder = std::thread::Builder::new().name(format!("mg: {}", name));

            let boundary_conditions = bt.map(|bt| match bt {
                euler::BoundaryCharacteristic::This => DistributedBoundaryConditions::This,
                euler::BoundaryCharacteristic::Grid(_) => DistributedBoundaryConditions::Channel,
                euler::BoundaryCharacteristic::Interpolate(_, int_op) => {
                    DistributedBoundaryConditions::Interpolate(int_op)
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

            let initial_conditions = self.initial_conditions.clone();

            tids.push(
                builder
                    .spawn(move || {
                        let (ny, nx) = (grid.ny(), grid.nx());
                        let mut current = Field::new(ny, nx);

                        match &initial_conditions {
                            parsing::InitialConditions::Vortex(vortexparams) => {
                                current.vortex(grid.x(), grid.y(), time, vortexparams)
                            }
                            parsing::InitialConditions::Expressions(expr) => {
                                // Evaluate the expressions on all variables
                                let x = grid.x();
                                let y = grid.y();
                                let (rho, rhou, rhov, e) = current.components_mut();
                                (*expr).evaluate(time, x, y, rho, rhou, rhov, e);
                            }
                        }
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
                                    .deflate(9)
                                    .shape((grid.ny(), grid.nx()))
                                    .create(name)
                            };
                            let xds = add_dim("x").unwrap();
                            xds.write(grid.x()).unwrap();
                            let yds = add_dim("y").unwrap();
                            yds.write(grid.y()).unwrap();

                            let add_var = |name| {
                                g.new_dataset::<Float>()
                                    .shuffle()
                                    .deflate(3)
                                    .chunk((1, grid.ny(), grid.nx()))
                                    .shape((0.., grid.ny(), grid.nx()))
                                    .create(name)
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
                            pull,
                            sbp,
                            t: time,
                            dt: Float::NAN,
                            initial_conditions,

                            _name: name,
                            id,

                            recv: t_recv,
                            send: master_send,

                            wb,
                            workbuffer_edges: {
                                Direction {
                                    north: (Array2::zeros((4, nx)), Array2::zeros((4, nx))),
                                    south: (Array2::zeros((4, nx)), Array2::zeros((4, nx))),
                                    east: (Array2::zeros((4, ny)), Array2::zeros((4, ny))),
                                    west: (Array2::zeros((4, ny)), Array2::zeros((4, ny))),
                                }
                            },
                            workbuffer_free: Direction {
                                north: {
                                    let mut arr = ArrayVec::new();
                                    for _ in 0..2 {
                                        arr.push(Array2::zeros((4, nx)))
                                    }
                                    arr
                                },
                                south: {
                                    let mut arr = ArrayVec::new();
                                    for _ in 0..2 {
                                        arr.push(Array2::zeros((4, nx)))
                                    }
                                    arr
                                },
                                east: {
                                    let mut arr = ArrayVec::new();
                                    for _ in 0..2 {
                                        arr.push(Array2::zeros((4, ny)))
                                    }
                                    arr
                                },
                                west: {
                                    let mut arr = ArrayVec::new();
                                    for _ in 0..2 {
                                        arr.push(Array2::zeros((4, ny)))
                                    }
                                    arr
                                },
                            },

                            progressbar: None,
                        };

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
            progressbar: None,
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
    pub fn add_progressbar(&mut self, ntime: u64) {
        match self {
            Self::SingleThreaded(sys) => sys.progressbar = Some(super::progressbar(ntime)),
            Self::MultiThreaded(sys) => sys.attach_progressbar(ntime),
        }
    }
    pub fn finish_progressbar(&mut self) {
        match self {
            Self::SingleThreaded(sys) => sys.progressbar.take().unwrap().finish_and_clear(),
            Self::MultiThreaded(sys) => {
                for tid in &sys.send {
                    tid.send(MsgFromHost::ProgressbarDrop).unwrap();
                }
                sys.synchronise();
                let target = sys.progressbar.take().unwrap();
                target.clear().unwrap();
            }
        }
    }
    pub fn error(&self) -> Float {
        match self {
            Self::SingleThreaded(sys) => sys.error(),
            Self::MultiThreaded(sys) => {
                for sender in &sys.send {
                    sender.send(MsgFromHost::Error).unwrap();
                }
                let mut e = 0.0;
                for _ in 0..sys.sys.len() {
                    e += match sys.recv.recv().unwrap() {
                        (_, MsgToHost::Error(e)) => e,
                        (_, m) => panic!("Unexpected message: {:?}", m),
                    }
                }
                e
            }
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
    pub progressbar: Option<indicatif::ProgressBar>,
    pub initial_conditions: parsing::InitialConditions,
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
            self.advance_single_step(self.dt);
            if let Some(pbar) = &self.progressbar {
                pbar.inc(1)
            }
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
        tds.write_slice(&[ntime], tpos..tpos + 1).unwrap();
        for (group, fnow) in self.output.1.iter().zip(&self.fnow) {
            let (ny, nx) = (fnow.ny(), fnow.nx());
            let rhods = group.dataset("rho").unwrap();
            let rhouds = group.dataset("rhou").unwrap();
            let rhovds = group.dataset("rhov").unwrap();
            let eds = group.dataset("e").unwrap();

            let (rho, rhou, rhov, e) = fnow.components();
            rhods.resize((tpos + 1, ny, nx)).unwrap();
            rhods.write_slice(rho, (tpos, .., ..)).unwrap();

            rhouds.resize((tpos + 1, ny, nx)).unwrap();
            rhouds.write_slice(rhou, (tpos, .., ..)).unwrap();

            rhovds.resize((tpos + 1, ny, nx)).unwrap();
            rhovds.write_slice(rhov, (tpos, .., ..)).unwrap();

            eds.resize((tpos + 1, ny, nx)).unwrap();
            eds.write_slice(e, (tpos, .., ..)).unwrap();
        }
    }

    pub fn error(&self) -> Float {
        let mut e = 0.0;
        for ((fmod, grid), op) in self.fnow.iter().zip(&self.grids).zip(&self.operators) {
            let mut fvort = fmod.clone();
            match &self.initial_conditions {
                parsing::InitialConditions::Vortex(vortexparams) => {
                    fvort.vortex(grid.x(), grid.y(), self.time, vortexparams);
                }
                parsing::InitialConditions::Expressions(expr) => {
                    let (rho, rhou, rhov, e) = fvort.components_mut();
                    expr.as_ref()
                        .evaluate(self.time, grid.x(), grid.y(), rho, rhou, rhov, e)
                }
            }
            e += fmod.h2_err(&fvort, &**op);
        }
        e
    }
}

pub struct DistributedSystem {
    recv: Receiver<(usize, MsgToHost)>,
    send: Vec<Sender<MsgFromHost>>,
    /// All threads should be joined to mark the end of the computation
    sys: Vec<std::thread::JoinHandle<()>>,
    output: hdf5::File,
    progressbar: Option<indicatif::MultiProgress>,
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
        tds.write_slice(&[ntime], tpos..tpos + 1).unwrap();
    }
    pub fn attach_progressbar(&mut self, ntime: u64) {
        let target = indicatif::MultiProgress::new();
        for tid in &self.send {
            let pb = super::progressbar(ntime);
            let pb = target.add(pb);
            tid.send(MsgFromHost::Progressbar(pb)).unwrap();
        }
        target.set_move_cursor(true);
        self.progressbar = Some(target);
    }
    fn send_barrier(&self, barrier: &crossbeam_utils::sync::WaitGroup) {
        for tid in &self.send {
            tid.send(MsgFromHost::Barrier(barrier.clone())).unwrap()
        }
    }
    pub fn synchronise(&self) {
        // Syncronise before starting the timer
        let barrier = crossbeam_utils::sync::WaitGroup::new();
        self.send_barrier(&barrier);
        barrier.wait();
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

/// Messages sent from the host to each compute thread
#[derive(Debug)]
enum MsgFromHost {
    /// Advance n steps
    Advance(u64),
    /// Compute the maximum dt allowed by this grid
    DtRequest,
    /// Set dt
    DtSet(Float),
    /// Output the current time to file
    Output(u64),
    /// Stop all computing
    Stop,
    /// Request the current error
    Error,
    /// A barrier that must be waited on
    Barrier(crossbeam_utils::sync::WaitGroup),
    /// Progressbar to report progress
    Progressbar(indicatif::ProgressBar),
    /// Clear and remove the progressbar
    ProgressbarDrop,
}

/// Messages sent back to the host
#[derive(Debug)]
enum MsgToHost {
    /// Maximum dt allowed by the current grid
    MaxDt(Float),
    /// Error from the current grid
    Error(Float),
}

// #[derive(Debug)]
pub enum DistributedBoundaryConditions {
    This,

    Vortex(VortexParameters),
    Eval(std::sync::Arc<dyn eval::Evaluator<ndarray::Ix1>>),

    Interpolate(Box<dyn InterpolationOperator>),
    Channel,
}

type CommunicatorData = ArrayVec<Array2<Float>, 4>;

struct Communicator {
    /// Waker for this grid, neighbours should have a reference
    /// and notify when a boundary has been put
    cvar: Condvar,
    /// Internal data exchange, is None on missing data, inner type
    /// can be set to None when grabbing the boundary
    data: Mutex<Direction<CommunicatorData>>,
}

struct DistributedSystemPart {
    grid: (Grid, Metrics),
    sbp: Box<dyn SbpOperator2d + 'static>,

    boundary_conditions: Direction<DistributedBoundaryConditions>,
    /// Channel pullers
    pull: Arc<Communicator>,
    /// Subscribers to the boundaries of self
    push: Direction<Option<Arc<Communicator>>>,

    current: Field,
    fut: Field,

    t: Float,
    dt: Float,

    _name: String,
    id: usize,
    recv: Receiver<MsgFromHost>,
    send: Sender<(usize, MsgToHost)>,

    output: hdf5::Group,
    initial_conditions: crate::parsing::InitialConditions,

    k: [Diff; 4],
    wb: WorkBuffers,
    /// Work buffer for boundaries
    workbuffer_edges: Direction<(Array2<Float>, Array2<Float>)>,
    /// These can be popped and pushed as we communicate data
    workbuffer_free: Direction<CommunicatorData>,

    progressbar: Option<indicatif::ProgressBar>,
}

impl DistributedSystemPart {
    fn run(&mut self) {
        loop {
            match self.recv.recv().unwrap() {
                MsgFromHost::DtSet(dt) => self.dt = dt,
                MsgFromHost::DtRequest => {
                    let dt = self.max_dt();
                    self.send(MsgToHost::MaxDt(dt)).unwrap();
                }
                MsgFromHost::Advance(ntime) => self.advance(ntime),
                MsgFromHost::Output(ntime) => self.output(ntime),
                MsgFromHost::Stop => return,
                MsgFromHost::Error => self.send(MsgToHost::Error(self.error())).unwrap(),
                MsgFromHost::Barrier(barrier) => barrier.wait(),
                MsgFromHost::Progressbar(pbar) => self.progressbar = Some(pbar),
                MsgFromHost::ProgressbarDrop => {
                    let pb = self.progressbar.take().unwrap();
                    pb.finish_and_clear()
                }
            }
        }
    }

    fn max_dt(&self) -> Float {
        let nx = self.current.nx();
        let ny = self.current.ny();

        let (rho, rhou, rhov, _e) = self.current.components();

        let mut max_u: Float = 0.0;
        let mut max_v: Float = 0.0;

        for ((((((rho, rhou), rhov), detj_dxi_dx), detj_dxi_dy), detj_deta_dx), detj_deta_dy) in rho
            .iter()
            .zip(rhou.iter())
            .zip(rhov.iter())
            .zip(self.grid.1.detj_dxi_dx())
            .zip(self.grid.1.detj_dxi_dy())
            .zip(self.grid.1.detj_deta_dx())
            .zip(self.grid.1.detj_deta_dy())
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

        let c_max = if self.sbp.is_h2xi() || self.sbp.is_h2eta() {
            0.5
        } else {
            1.0
        };

        c_max / c_dt
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
        rhods.write_slice(rho, (tpos, .., ..)).unwrap();

        rhouds.resize((tpos + 1, ny, nx)).unwrap();
        rhouds.write_slice(rhou, (tpos, .., ..)).unwrap();

        rhovds.resize((tpos + 1, ny, nx)).unwrap();
        rhovds.write_slice(rhov, (tpos, .., ..)).unwrap();

        eds.resize((tpos + 1, ny, nx)).unwrap();
        eds.write_slice(e, (tpos, .., ..)).unwrap();
    }

    fn advance(&mut self, ntime: u64) {
        for _itime in 0..ntime {
            if let Some(pbar) = &self.progressbar {
                pbar.inc(1)
            }
            let metrics = &self.grid.1;
            let wb = &mut self.wb.0;
            let sbp = &self.sbp;
            let push = &self.push;
            let pull = &self.pull;
            let boundary_conditions = &self.boundary_conditions;
            let grid = &self.grid.0;
            let workbuffer_edges = &mut self.workbuffer_edges;
            let workbuffer_free = &mut self.workbuffer_free;

            let rhs = |k: &mut euler::Diff, y: &euler::Field, time: Float| {
                // Send off the boundaries eagerly, in case neighbouring grid is ready
                push.as_ref()
                    .zip(workbuffer_free.as_mut())
                    .zip(
                        Direction::<fn(&mut Direction<CommunicatorData>) -> &mut CommunicatorData> {
                            north: |x| x.south_mut(),
                            south: |x| x.north_mut(),
                            east: |x| x.west_mut(),
                            west: |x| x.east_mut(),
                        },
                    )
                    .zip(Direction {
                        north: y.north(),
                        south: y.south(),
                        west: y.west(),
                        east: y.east(),
                    })
                    .map(|(((push, wb), sel), this)| {
                        if let Some(s) = push {
                            let mut wb = wb.pop().unwrap();
                            wb.assign(&this);
                            {
                                let mut s = s.data.lock();
                                sel(&mut s).push(wb);
                            }
                            s.cvar.notify_one();
                        }
                    });

                // This computation does not depend on the boundaries
                euler::RHS_no_SAT(sbp.deref(), k, y, metrics, wb);

                // Get boundaries, but be careful and maximise the amount of work which can be
                // performed before we have all of them, whilst ensuring threads can sleep for as
                // long as possible
                let computed = boundary_conditions
                    .as_ref()
                    .zip(euler::SAT_FUNCTIONS)
                    .zip(workbuffer_edges.as_mut())
                    .zip(Direction {
                        north: y.south(),
                        south: y.north(),
                        east: y.west(),
                        west: y.east(),
                    })
                    .zip(Direction {
                        north: grid.north(),
                        south: grid.south(),
                        east: grid.east(),
                        west: grid.west(),
                    })
                    .map(|((((bc, sat), wb), self_edge), grid)| {
                        wb.0.fill(0.0);
                        match bc {
                            DistributedBoundaryConditions::Channel
                            | DistributedBoundaryConditions::Interpolate(_) => false,
                            DistributedBoundaryConditions::This => {
                                sat(sbp.deref(), wb.0.view_mut(), y, metrics, self_edge);
                                true
                            }
                            DistributedBoundaryConditions::Vortex(vp) => {
                                let mut fiter = wb.1.outer_iter_mut();
                                let (rho, rhou, rhov, e) = (
                                    fiter.next().unwrap(),
                                    fiter.next().unwrap(),
                                    fiter.next().unwrap(),
                                    fiter.next().unwrap(),
                                );
                                let (gx, gy) = grid;
                                vp.evaluate(time, gx, gy, rho, rhou, rhov, e);

                                sat(sbp.deref(), wb.0.view_mut(), y, metrics, wb.1.view());
                                true
                            }
                            DistributedBoundaryConditions::Eval(eval) => {
                                let mut fiter = wb.1.outer_iter_mut();
                                let (rho, rhou, rhov, e) = (
                                    fiter.next().unwrap(),
                                    fiter.next().unwrap(),
                                    fiter.next().unwrap(),
                                    fiter.next().unwrap(),
                                );
                                let (gx, gy) = grid;
                                eval.evaluate(time, gx, gy, rho, rhou, rhov, e);
                                sat(sbp.deref(), wb.0.view_mut(), y, metrics, wb.1.view());
                                true
                            }
                        }
                    });

                if computed.north {
                    k.north_mut()
                        .scaled_add(1.0, &workbuffer_edges.north().0.view());
                }
                if computed.south {
                    k.south_mut()
                        .scaled_add(1.0, &workbuffer_edges.south().0.view());
                }
                if computed.east {
                    k.east_mut()
                        .scaled_add(1.0, &workbuffer_edges.east().0.view());
                }
                if computed.west {
                    k.west_mut()
                        .scaled_add(1.0, &workbuffer_edges.west().0.view());
                }

                let mut boundaries_remaining = computed.map(|b| !b);

                {
                    let mut data = pull.data.lock();
                    'check_boundaries: while boundaries_remaining.any() {
                        let boundaries =
                            boundaries_remaining
                                .zip(data.as_mut())
                                .map(
                                    |(remains, data)| {
                                        if remains {
                                            data.pop_at(0)
                                        } else {
                                            None
                                        }
                                    },
                                );
                        if boundaries.as_ref().map(Option::is_none).all() {
                            // Park thread while waiting for boundaries
                            pull.cvar.wait(&mut data);
                            continue 'check_boundaries;
                        }
                        // While we are waiting we can unlock mutex
                        lock_api::MutexGuard::unlocked(&mut data, || {
                            if let Some(mut boundary) = boundaries.north {
                                boundaries_remaining.north = false;
                                let wb = workbuffer_edges.north_mut();
                                let wb_push = workbuffer_free.north_mut();
                                match boundary_conditions.north() {
                                    DistributedBoundaryConditions::Channel => {
                                        std::mem::swap(&mut wb.0, &mut boundary);
                                        wb_push.push(boundary);
                                    }
                                    DistributedBoundaryConditions::Interpolate(int_op) => {
                                        let is_fine2coarse = boundary.shape()[1] > wb.0.shape()[2];
                                        for (to, from) in
                                            wb.0.outer_iter_mut().zip(boundary.outer_iter())
                                        {
                                            if is_fine2coarse {
                                                int_op.fine2coarse(from, to);
                                            } else {
                                                int_op.coarse2fine(from, to);
                                            }
                                        }
                                        // Reshape edge buffer to correct size
                                        let mut vec = boundary.into_raw_vec();
                                        vec.resize(wb.0.len(), 0.0);
                                        let boundary =
                                            Array2::from_shape_vec(wb.0.raw_dim(), vec).unwrap();
                                        wb_push.push(boundary)
                                    }
                                    _ => unreachable!(),
                                }
                                euler::SAT_north(
                                    sbp.deref(),
                                    k.north_mut(),
                                    y,
                                    metrics,
                                    wb.0.view(),
                                );
                            };

                            if let Some(mut boundary) = boundaries.south {
                                boundaries_remaining.south = false;
                                let wb = workbuffer_edges.north_mut();
                                let wb_push = workbuffer_free.south_mut();
                                match boundary_conditions.south() {
                                    DistributedBoundaryConditions::Channel => {
                                        std::mem::swap(&mut wb.0, &mut boundary);
                                        wb_push.push(boundary);
                                    }
                                    DistributedBoundaryConditions::Interpolate(int_op) => {
                                        let is_fine2coarse = boundary.shape()[1] > wb.0.shape()[2];
                                        for (to, from) in
                                            wb.0.outer_iter_mut().zip(boundary.outer_iter())
                                        {
                                            if is_fine2coarse {
                                                int_op.fine2coarse(from, to);
                                            } else {
                                                int_op.coarse2fine(from, to);
                                            }
                                        }
                                        // Reshape edge buffer to correct size
                                        let mut vec = boundary.into_raw_vec();
                                        vec.resize(wb.0.len(), 0.0);
                                        let boundary =
                                            Array2::from_shape_vec(wb.0.raw_dim(), vec).unwrap();
                                        wb_push.push(boundary);
                                    }
                                    _ => unreachable!(),
                                }
                                euler::SAT_south(
                                    sbp.deref(),
                                    k.south_mut(),
                                    y,
                                    metrics,
                                    wb.0.view(),
                                );
                            };

                            if let Some(mut boundary) = boundaries.east {
                                boundaries_remaining.east = false;
                                let wb = workbuffer_edges.east_mut();
                                let wb_push = workbuffer_free.east_mut();
                                match boundary_conditions.east() {
                                    DistributedBoundaryConditions::Channel => {
                                        std::mem::swap(&mut wb.0, &mut boundary);
                                        wb_push.push(boundary);
                                    }
                                    DistributedBoundaryConditions::Interpolate(int_op) => {
                                        let is_fine2coarse = boundary.shape()[1] > wb.0.shape()[2];
                                        for (to, from) in
                                            wb.0.outer_iter_mut().zip(boundary.outer_iter())
                                        {
                                            if is_fine2coarse {
                                                int_op.fine2coarse(from, to);
                                            } else {
                                                int_op.coarse2fine(from, to);
                                            }
                                        }
                                        // Reshape edge buffer to correct size
                                        let mut vec = boundary.into_raw_vec();
                                        vec.resize(wb.0.len(), 0.0);
                                        let boundary =
                                            Array2::from_shape_vec(wb.0.raw_dim(), vec).unwrap();
                                        wb_push.push(boundary);
                                    }
                                    _ => unreachable!(),
                                }
                                euler::SAT_east(sbp.deref(), k.east_mut(), y, metrics, wb.0.view());
                            };

                            if let Some(mut boundary) = boundaries.west {
                                boundaries_remaining.west = false;
                                let wb = workbuffer_edges.west_mut();
                                let wb_push = workbuffer_free.west_mut();
                                match boundary_conditions.west() {
                                    DistributedBoundaryConditions::Channel => {
                                        std::mem::swap(&mut wb.0, &mut boundary);
                                        wb_push.push(boundary);
                                    }
                                    DistributedBoundaryConditions::Interpolate(int_op) => {
                                        let is_fine2coarse = boundary.shape()[1] > wb.0.shape()[2];
                                        for (to, from) in
                                            wb.0.outer_iter_mut().zip(boundary.outer_iter())
                                        {
                                            if is_fine2coarse {
                                                int_op.fine2coarse(from, to);
                                            } else {
                                                int_op.coarse2fine(from, to);
                                            }
                                        }
                                        // Reshape edge buffer to correct size
                                        let mut vec = boundary.into_raw_vec();
                                        vec.resize(wb.0.len(), 0.0);
                                        let boundary =
                                            Array2::from_shape_vec(wb.0.raw_dim(), vec).unwrap();
                                        wb_push.push(boundary);
                                    }
                                    _ => unreachable!(),
                                }
                                euler::SAT_west(sbp.deref(), k.west_mut(), y, metrics, wb.0.view());
                            };
                        });
                    }
                }
            };
            integrate::integrate::<integrate::Rk4, Field, _>(
                rhs,
                &self.current,
                &mut self.fut,
                &mut self.t,
                self.dt,
                &mut self.k,
            );
            std::mem::swap(&mut self.current, &mut self.fut);
        }
    }

    fn send(&self, msg: MsgToHost) -> Result<(), crossbeam_channel::SendError<(usize, MsgToHost)>> {
        self.send.send((self.id, msg))
    }

    fn error(&self) -> Float {
        let mut fvort = self.current.clone();
        match &self.initial_conditions {
            parsing::InitialConditions::Vortex(vortexparams) => {
                fvort.vortex(self.grid.0.x(), self.grid.0.y(), self.t, vortexparams);
            }
            parsing::InitialConditions::Expressions(expr) => {
                let (rho, rhou, rhov, e) = fvort.components_mut();
                expr.as_ref()
                    .evaluate(self.t, self.grid.0.x(), self.grid.0.y(), rho, rhou, rhov, e)
            }
        }
        self.current.h2_err(&fvort, &*self.sbp)
    }
}
