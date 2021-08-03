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
    pub fn distribute(self, ntime: usize) -> DistributedSystem {
        let nthreads = self.grids.len();
        let time = 0.0;
        // alt: crossbeam::WaitGroup
        let b = Arc::new(Barrier::new(nthreads + 1));
        let dt = self.max_dt();

        // Build up the boundary conditions
        // Assume all boundaries are push/pull through channels
        let channels = (0..nthreads)
            .map(|_| {
                use crossbeam_channel::unbounded;
                Direction {
                    north: unbounded(),
                    south: unbounded(),
                    west: unbounded(),
                    east: unbounded(),
                }
            })
            .collect::<Vec<_>>();

        // TODO: Iterate through all grids and see if they need ourself to push
        let mut requested_channels = (0..nthreads)
            .map(|_| Direction::default())
            .collect::<Vec<Direction<bool>>>();

        let mut tids = Vec::new();
        for (((((((current, fut), grid), metrics), sbp), wb), bt), req_channel) in self
            .fnow
            .into_iter()
            .zip(self.fnext.into_iter())
            .zip(self.grids.into_iter())
            .zip(self.metrics.into_iter())
            .zip(self.operators.into_iter())
            .zip(self.wb.into_iter())
            .zip(self.bt)
            .zip(requested_channels)
        {
            let builder = std::thread::Builder::new().name(format!("eulersolver: {}", "smth"));
            let barrier = b.clone();

            let Direction {
                north: bt_north,
                south: bt_south,
                west: bt_west,
                east: bt_east,
            } = bt;
            let boundary_conditions = Direction {
                north: match bt_north {
                    euler::BoundaryCharacteristic::This => DistributedBoundaryConditions::This,
                    euler::BoundaryCharacteristic::Grid(i) => {
                        *requested_channels[i].south_mut() = true;
                        DistributedBoundaryConditions::Channel(channels[i].south().1.clone())
                    }
                    euler::BoundaryCharacteristic::Interpolate(i, int_op) => {
                        *requested_channels[i].south_mut() = true;
                        DistributedBoundaryConditions::Interpolate(
                            channels[i].south().1.clone(),
                            int_op,
                        )
                    }
                    euler::BoundaryCharacteristic::MultiGrid(_) => unimplemented!(),
                    euler::BoundaryCharacteristic::Vortex(vp) => {
                        DistributedBoundaryConditions::Vortex(vp)
                    }
                    euler::BoundaryCharacteristic::Eval(eval) => {
                        DistributedBoundaryConditions::Eval(eval)
                    }
                },
                south: match bt_south {
                    euler::BoundaryCharacteristic::This => DistributedBoundaryConditions::This,
                    euler::BoundaryCharacteristic::Grid(i) => {
                        *requested_channels[i].north_mut() = true;
                        DistributedBoundaryConditions::Channel(channels[i].north().1.clone())
                    }
                    euler::BoundaryCharacteristic::Interpolate(i, int_op) => {
                        *requested_channels[i].north_mut() = true;
                        DistributedBoundaryConditions::Interpolate(
                            channels[i].north().1.clone(),
                            int_op,
                        )
                    }
                    euler::BoundaryCharacteristic::MultiGrid(_) => unimplemented!(),
                    euler::BoundaryCharacteristic::Vortex(vp) => {
                        DistributedBoundaryConditions::Vortex(vp)
                    }
                    euler::BoundaryCharacteristic::Eval(eval) => {
                        DistributedBoundaryConditions::Eval(eval)
                    }
                },
                east: match bt_east {
                    euler::BoundaryCharacteristic::This => DistributedBoundaryConditions::This,
                    euler::BoundaryCharacteristic::Grid(i) => {
                        *requested_channels[i].west_mut() = true;
                        DistributedBoundaryConditions::Channel(channels[i].west().1.clone())
                    }
                    euler::BoundaryCharacteristic::Interpolate(i, int_op) => {
                        *requested_channels[i].west_mut() = true;
                        DistributedBoundaryConditions::Interpolate(
                            channels[i].west().1.clone(),
                            int_op,
                        )
                    }
                    euler::BoundaryCharacteristic::MultiGrid(_) => unimplemented!(),
                    euler::BoundaryCharacteristic::Vortex(vp) => {
                        DistributedBoundaryConditions::Vortex(vp)
                    }
                    euler::BoundaryCharacteristic::Eval(eval) => {
                        DistributedBoundaryConditions::Eval(eval)
                    }
                },
                west: match bt_west {
                    euler::BoundaryCharacteristic::This => DistributedBoundaryConditions::This,
                    euler::BoundaryCharacteristic::Grid(i) => {
                        *requested_channels[i].east_mut() = true;
                        DistributedBoundaryConditions::Channel(channels[i].east().1.clone())
                    }
                    euler::BoundaryCharacteristic::Interpolate(i, int_op) => {
                        *requested_channels[i].east_mut() = true;
                        DistributedBoundaryConditions::Interpolate(
                            channels[i].east().1.clone(),
                            int_op,
                        )
                    }
                    euler::BoundaryCharacteristic::MultiGrid(_) => unimplemented!(),
                    euler::BoundaryCharacteristic::Vortex(vp) => {
                        DistributedBoundaryConditions::Vortex(vp)
                    }
                    euler::BoundaryCharacteristic::Eval(eval) => {
                        DistributedBoundaryConditions::Eval(eval)
                    }
                },
            };

            tids.push(
                builder
                    .spawn(move || {
                        let mut sys = DistributedSystemPart {
                            barrier,
                            ntime,
                            dt,
                            current,
                            fut,
                            k: [todo!(); 4],
                            boundary_conditions,
                            grid: (grid, metrics),
                            output: (),
                            push: todo!(),
                            sbp,
                            t: time,
                            wb,
                        };
                        sys.advance();
                    })
                    .unwrap(),
            );
        }
        // Set up communicators
        // Spawn a new communicator

        DistributedSystem {
            ntime,
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
    ntime: usize,
    /// These should be joined to mark the end of the computation
    sys: Vec<std::thread::JoinHandle<()>>,
}

impl DistributedSystem {
    fn run(self) {
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

#[derive(Debug, Clone)]
enum PushCommunicator {
    Channel(Sender<Array2<Float>>),
    None,
}

impl Default for PushCommunicator {
    fn default() -> Self {
        Self::None
    }
}

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
    output: (), // hdf5::Dataset eventually,
    t: Float,
    dt: Float,
    ntime: usize,
}

impl DistributedSystemPart {
    fn advance(&mut self) {
        self.barrier.wait();
        for i in 0..self.ntime {
            let metrics = &self.grid.1;
            let wb = &mut self.wb.0;
            let sbp = &self.sbp;
            let push = &self.push;
            let boundary_conditions = &self.boundary_conditions;
            let grid = &self.grid.0;

            let mut rhs = |k: &mut euler::Diff, y: &euler::Field, time: Float| {
                // Send off the boundaries optimistically, in case some grid is ready
                match &push.north {
                    PushCommunicator::None => (),
                    PushCommunicator::Channel(s) => s.send(y.north().to_owned()).unwrap(),
                }
                match &push.south {
                    PushCommunicator::None => (),
                    PushCommunicator::Channel(s) => s.send(y.south().to_owned()).unwrap(),
                }
                match &push.east {
                    PushCommunicator::None => (),
                    PushCommunicator::Channel(s) => s.send(y.east().to_owned()).unwrap(),
                }
                match &push.west {
                    PushCommunicator::None => (),
                    PushCommunicator::Channel(s) => s.send(y.west().to_owned()).unwrap(),
                }

                use std::ops::Deref;
                // This computation does not depend on the boundaries
                euler::RHS_no_SAT(sbp.deref(), k, y, metrics, wb);

                fn north_sat() {
                    todo!()
                }

                // Get boundaries, but be careful and maximise the amount of work which can be
                // performed before we have all of them, whilst ensuring threads can sleep for as
                // long as possible
                let mut select = Select::new();
                let mut selectable = 0;
                let recv_north = match boundary_conditions.north() {
                    DistributedBoundaryConditions::Channel(r)
                    | DistributedBoundaryConditions::Interpolate(r, _) => Some(r),
                    DistributedBoundaryConditions::This => {
                        todo!()
                    }
                    DistributedBoundaryConditions::Vortex(vp) => {
                        let mut data = y.north().to_owned();
                        let mut fiter = data.outer_iter_mut();
                        let (rho, rhou, rhov, e) = (
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                            fiter.next().unwrap(),
                        );
                        let (x, y) = grid.north();
                        vp.evaluate(time, x, y, rho, rhou, rhov, e);

                        north_sat();
                        None
                    }
                    DistributedBoundaryConditions::Eval(eval) => {
                        todo!()
                    }
                    _ => None,
                };
                let recv_south = if let Some(r) = boundary_conditions.south().channel() {
                    selectable += 1;
                    Some(select.recv(r))
                } else {
                    // Do SAT boundary from other BC
                    None
                };
                let recv_west = if let Some(r) = boundary_conditions.west().channel() {
                    selectable += 1;
                    Some(select.recv(r))
                } else {
                    // Do SAT boundary from other BC
                    None
                };
                let recv_east = if let Some(r) = boundary_conditions.east().channel() {
                    selectable += 1;
                    Some(select.recv(r))
                } else {
                    // Do SAT boundary from other BC
                    None
                };

                // Get an item off each channel, waiting minimally before processing that boundary.
                // The waiting ensures other grids can be processed by the core in case of
                // oversubscription (in case of a more grids than core scenario)
                // This minimises the amount of time waiting on boundary conditions
                while selectable != 0 {
                    let s = select.select();
                    let sindex = s.index();
                    match Some(sindex) {
                        recv_north => {
                            let r = s.recv(boundary_conditions.north().channel().unwrap());
                            // process into boundary SAT here
                        }
                        recv_south => {
                            let r = s.recv(boundary_conditions.south().channel().unwrap());
                            // process into boundary SAT here
                        }
                        recv_west => {
                            let r = s.recv(boundary_conditions.west().channel().unwrap());
                            // process into boundary SAT here
                        }
                        recv_east => {
                            let r = s.recv(boundary_conditions.east().channel().unwrap());
                            // process into boundary SAT here
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
