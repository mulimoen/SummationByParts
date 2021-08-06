use crate::parsing;
use crate::utils::Direction;
use crossbeam_channel::{Receiver, Select, Sender};
use euler::{
    eval::{self, Evaluator},
    Diff, Field, VortexParameters, WorkBuffers,
};
use ndarray::Array2;
use sbp::grid::{Grid, Metrics};
use sbp::operators::{InterpolationOperator, SbpOperator2d};
use sbp::*;
use std::sync::{Arc, Barrier};

pub struct BaseSystem {
    pub names: Vec<String>,
    pub grids: Vec<grid::Grid>,
    pub time: Float,
    pub boundary_conditions: Vec<euler::BoundaryCharacteristics>,
    pub initial_conditions: crate::parsing::InitialConditions,
    pub operators: Vec<Box<dyn SbpOperator2d>>,
}

impl BaseSystem {
    pub fn new(
        names: Vec<String>,
        grids: Vec<grid::Grid>,
        time: Float,
        operators: Vec<Box<dyn SbpOperator2d>>,
        boundary_conditions: Vec<euler::BoundaryCharacteristics>,
        initial_conditions: crate::parsing::InitialConditions,
    ) -> Self {
        Self {
            names,
            grids,
            time,
            boundary_conditions,
            initial_conditions,
            operators,
        }
    }
    pub fn create(self) -> System {
        let mut sys = System::new(self.grids, self.boundary_conditions, self.operators);
        match &self.initial_conditions {
            /*
            parsing::InitialConditions::File(f) => {
                for grid in &sys.grids {
                    // Copy initial conditions from file, requires name of field
                    todo!()
                }
            }
            */
            parsing::InitialConditions::Vortex(vortexparams) => sys.vortex(0.0, &vortexparams),
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
        sys
    }
}

pub struct System {
    pub fnow: Vec<euler::Field>,
    pub fnext: Vec<euler::Field>,
    pub wb: Vec<euler::WorkBuffers>,
    pub k: [Vec<euler::Diff>; 4],
    pub grids: Vec<grid::Grid>,
    pub metrics: Vec<grid::Metrics>,
    pub bt: Vec<euler::BoundaryCharacteristics>,
    pub eb: Vec<euler::BoundaryStorage>,
    pub time: Float,
    pub operators: Vec<Box<dyn SbpOperator2d>>,
}

impl integrate::Integrable for System {
    type State = Vec<euler::Field>;
    type Diff = Vec<euler::Diff>;

    fn scaled_add(s: &mut Self::State, o: &Self::Diff, scale: Float) {
        s.iter_mut()
            .zip(o.iter())
            .for_each(|(s, o)| euler::Field::scaled_add(s, o, scale))
    }
}

impl System {
    pub fn new(
        grids: Vec<grid::Grid>,
        bt: Vec<euler::BoundaryCharacteristics>,
        operators: Vec<Box<dyn SbpOperator2d>>,
    ) -> Self {
        let fnow = grids
            .iter()
            .map(|g| euler::Field::new(g.ny(), g.nx()))
            .collect::<Vec<_>>();
        let fnext = fnow.clone();
        let wb = grids
            .iter()
            .map(|g| euler::WorkBuffers::new(g.ny(), g.nx()))
            .collect();
        let k = grids
            .iter()
            .map(|g| euler::Diff::zeros((g.ny(), g.nx())))
            .collect::<Vec<_>>();
        let k = [k.clone(), k.clone(), k.clone(), k];
        let metrics = grids
            .iter()
            .zip(&operators)
            .map(|(g, op)| g.metrics(&**op).unwrap())
            .collect::<Vec<_>>();

        let eb = bt
            .iter()
            .zip(&grids)
            .map(|(bt, grid)| euler::BoundaryStorage::new(bt, grid))
            .collect();

        Self {
            fnow,
            fnext,
            k,
            wb,
            grids,
            metrics,
            bt,
            eb,
            time: 0.0,
            operators,
        }
    }

    pub fn vortex(&mut self, t: Float, vortex_params: &euler::VortexParameters) {
        for (f, g) in self.fnow.iter_mut().zip(&self.grids) {
            f.vortex(g.x(), g.y(), t, &vortex_params);
        }
    }

    pub fn advance(&mut self, dt: Float) {
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

        integrate::integrate::<integrate::Rk4, System, _>(
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

    /// Spreads the computation over n threads, in a thread per grid way.
    /// This system can only be called once for ntime calls.
    pub fn distribute(self, ntime: u64) -> DistributedSystem {
        let nthreads = self.grids.len();
        let time = 0.0;
        // alt: crossbeam::WaitGroup
        let b = Arc::new(Barrier::new(nthreads + 1));
        let dt = self.max_dt();

        // Build up the boundary conditions
        let mut push_channels: Vec<Direction<Option<Sender<Array2<Float>>>>> =
            Vec::with_capacity(nthreads);
        let mut pull_channels: Vec<Direction<Option<Receiver<Array2<Float>>>>> =
            vec![Direction::default(); nthreads];

        // Build the set of communicators between boundaries
        for wb in &self.bt {
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

        let mut tids = Vec::new();
        for ((((((((current, fut), grid), metrics), sbp), wb), bt), chan), push) in self
            .fnow
            .into_iter()
            .zip(self.fnext.into_iter())
            .zip(self.grids.into_iter())
            .zip(self.metrics.into_iter())
            .zip(self.operators.into_iter())
            .zip(self.wb.into_iter())
            .zip(self.bt)
            .zip(pull_channels)
            .zip(push_channels)
        {
            let builder = std::thread::Builder::new().name(format!("eulersolver: {}", "smth"));
            let barrier = b.clone();

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

            let (ny, nx) = (grid.nx(), grid.ny());

            tids.push(
                builder
                    .spawn(move || {
                        let mut sys = DistributedSystemPart {
                            barrier,
                            ntime,
                            dt,
                            current,
                            fut,
                            k: [
                                Diff::zeros((ny, nx)),
                                Diff::zeros((ny, nx)),
                                Diff::zeros((ny, nx)),
                                Diff::zeros((ny, nx)),
                            ],
                            boundary_conditions,
                            grid: (grid, metrics),
                            _output: (),
                            push,
                            sbp,
                            t: time,

                            wb,
                            wb_ns: Array2::zeros((4, nx)),
                            wb_ew: Array2::zeros((4, ny)),
                        };
                        sys.advance();
                    })
                    .unwrap(),
            );
        }
        // Set up communicators
        // Spawn a new communicator

        DistributedSystem {
            _ntime: ntime,
            start: b,
            sys: tids,
        }
    }
}

// single-threaded items: clone_from
// sync points: scaled_add, rhs
//
// Could instead make every thread (of a threadpool?) carry one grid themselves,
// and instead only wait on obtaining bc, which would make synchronisation
// be somewhat minimal
//
// Difficulties/implementation notes:
// * How to get output from each thread? Push to the multi-producer, single consumer queue
//   with capacity of 2*N (which ensures internal synchronisation stays active), or many
//   channels which are select!'ed. Choose instead a (name, itime, data) tuple, which is
//   communicated. Also initialise the output file all at the start, since we know the
//   number of time steps, and which timesteps will be used (should_output).
// * How to balance the number of threads/jobs?
//   Each grid will be pushed to a thread, and a thread pool created for use by all threads
//   combined. Set affinity to limit the number of cores available (could have side effects
//   on heat etc, although this would also be the case for multi-threaded code). A
//   thread-per-core architecture + the output thread if we have enough cores, else let os
//   control the scheduling for us.
//   Use cgroups to artificially limit the number of cores.
//   Use taskset to artificially limit the number of cores.
// * Mechanism to wait for bc, is select available? Use a channel with 0-4 capacity, each
//   other grid additonally pushes their direction to always use the available
//   Use select or similar from crossbeam_channel
// * Mechanism to push bc: try_push and then move to next bc if not available. Spawn a
//   thread into a threadpool to handle this.
// * Each grid is spawned onto an async threadpool with M threads allocated to itself, this should
//   allow async waiting on the channel (or in the future from process/internet/bc processor)
// * BC: A local thread can have a handle
pub struct DistributedSystem {
    /// Simple messaging system to be replaced by a more sophisticated system (e.g. run 5 steps,
    /// collect, initialise, return something to main)
    /// This simply waits until all threads are ready, then starts the computation on all threads
    start: Arc<Barrier>,
    _ntime: u64,
    /// These should be joined to mark the end of the computation
    sys: Vec<std::thread::JoinHandle<()>>,
}

impl DistributedSystem {
    pub fn run(self) {
        // This should start as we have n thread, but barrier blocks on n+1
        self.start.wait();
        self.sys.into_iter().for_each(|tid| tid.join().unwrap());
    }
}

// #[derive(Debug)]
pub enum DistributedBoundaryConditions {
    This,

    Vortex(VortexParameters),
    Eval(std::sync::Arc<dyn eval::Evaluator<ndarray::Ix1>>),

    Interpolate(Receiver<Array2<Float>>, Box<dyn InterpolationOperator>),
    Channel(Receiver<Array2<Float>>),
}

impl DistributedBoundaryConditions {
    fn channel(&self) -> Option<&Receiver<Array2<Float>>> {
        match self {
            Self::Interpolate(r, _) | Self::Channel(r) => Some(r),
            _ => None,
        }
    }
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
    k: [Diff; 4],
    wb: WorkBuffers,
    barrier: Arc<Barrier>,
    _output: (), // hdf5::Dataset eventually,
    t: Float,
    dt: Float,
    ntime: u64,

    /// Work buffer for boundaries
    wb_ns: Array2<Float>,
    wb_ew: Array2<Float>,
}

impl DistributedSystemPart {
    fn advance(&mut self) {
        self.barrier.wait();
        for _i in 0..self.ntime {
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

                use std::ops::Deref;
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
                        x if x == recv_north => {
                            let r = s
                                .recv(boundary_conditions.north().channel().unwrap())
                                .unwrap();
                            // TODO: Interpolation
                            euler::SAT_north(sbp.deref(), k, y, metrics, r.view());
                        }
                        x if x == recv_south => {
                            let r = s
                                .recv(boundary_conditions.south().channel().unwrap())
                                .unwrap();
                            // TODO: Interpolation
                            euler::SAT_south(sbp.deref(), k, y, metrics, r.view());
                        }
                        x if x == recv_west => {
                            let r = s
                                .recv(boundary_conditions.west().channel().unwrap())
                                .unwrap();
                            // TODO: Interpolation
                            euler::SAT_west(sbp.deref(), k, y, metrics, r.view());
                        }
                        x if x == recv_east => {
                            let r = s
                                .recv(boundary_conditions.east().channel().unwrap())
                                .unwrap();
                            // TODO: Interpolation
                            euler::SAT_east(sbp.deref(), k, y, metrics, r.view());
                        }
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
