use std::convert::{TryFrom, TryInto};

use sbp::operators::SbpOperator2d;
use sbp::Float;

use crate::eval;
use crate::input;

impl TryFrom<input::InitialConditions> for InitialConditions {
    type Error = ();
    fn try_from(v: input::InitialConditions) -> Result<Self, Self::Error> {
        Ok(match v {
            input::InitialConditions::Vortex(v) => Self::Vortex(v),
            // InputInitialConditions::File(file) => Self::File(hdf5::File::open(file).unwrap()),
            input::InitialConditions::Expressions(expr) => {
                Self::Expressions(std::sync::Arc::new(expr.try_into()?))
            }
        })
    }
}

#[derive(Clone, Debug)]
pub enum InitialConditions {
    Vortex(euler::VortexParameters),
    // File(hdf5::File),
    Expressions(std::sync::Arc<eval::Evaluator>),
}

#[derive(Clone, Debug)]
enum BoundaryConditions {
    Vortex(euler::VortexParameters),
    Expressions(std::sync::Arc<eval::Evaluator>),
    NotNeeded,
}

pub struct RuntimeConfiguration {
    pub names: Vec<String>,
    pub grids: Vec<sbp::grid::Grid>,
    pub boundary_conditions: Vec<euler::BoundaryCharacteristics>,
    pub op: Vec<Box<dyn SbpOperator2d>>,
    pub integration_time: Float,
    pub initial_conditions: InitialConditions,
}

impl input::Configuration {
    pub fn into_runtime(mut self) -> RuntimeConfiguration {
        let gamma = self.gamma;
        let _ = euler::GAMMA.set(gamma);
        let default = self.grids.shift_remove("default").unwrap_or_default();
        let names = self.grids.keys().cloned().collect();

        let initial_conditions: InitialConditions =
            self.initial_conditions.clone().try_into().unwrap();

        let boundary_conditions = match &self.boundary_conditions {
            input::BoundaryConditions::NotNeeded => BoundaryConditions::NotNeeded,
            input::BoundaryConditions::Vortex(vp) => BoundaryConditions::Vortex(vp.clone()),
            input::BoundaryConditions::Expressions(expr) => BoundaryConditions::Expressions(
                std::sync::Arc::new(expr.clone().try_into().unwrap()),
            ),
            input::BoundaryConditions::InputInitialConditions => match &initial_conditions {
                InitialConditions::Vortex(vp) => BoundaryConditions::Vortex(vp.clone()),
                InitialConditions::Expressions(expr) => {
                    BoundaryConditions::Expressions(expr.clone())
                } // _ => panic!("Boundary conditions were set to initial conditions, although initial conditions are not available",),
            },
        };

        let grids = self
            .grids
            .iter()
            .map(|(_name, g)| {
                use input::ArrayForm;
                let x: ArrayForm =
                    g.x.clone()
                        .unwrap_or_else(|| default.x.as_ref().unwrap().clone())
                        .into();
                let y: ArrayForm =
                    g.y.clone()
                        .unwrap_or_else(|| default.y.as_ref().unwrap().clone())
                        .into();
                let (x, y) = match (x, y) {
                    (ArrayForm::Array1(x), ArrayForm::Array1(y)) => {
                        let xlen = x.len();
                        let ylen = y.len();
                        let x = x.broadcast((ylen, xlen)).unwrap().to_owned();
                        let y = y
                            .broadcast((xlen, ylen))
                            .unwrap()
                            .reversed_axes()
                            .to_owned();

                        (x, y)
                    }
                    (ArrayForm::Array1(x), ArrayForm::Array2(y)) => {
                        assert_eq!(x.len(), y.shape()[1]);
                        let x = x.broadcast((y.shape()[1], x.len())).unwrap().to_owned();
                        (x, y)
                    }
                    (ArrayForm::Array2(x), ArrayForm::Array1(y)) => {
                        assert_eq!(x.shape()[0], y.len());
                        let y = y
                            .broadcast((x.shape()[1], y.len()))
                            .unwrap()
                            .reversed_axes()
                            .to_owned();
                        (x, y)
                    }
                    (ArrayForm::Array2(x), ArrayForm::Array2(y)) => {
                        assert_eq!(x.shape(), y.shape());
                        (x, y)
                    } /*
                      (ArrayForm::Unknown, ArrayForm::Unknown) => {
                          if let InitialConditions::File(file) = &initial_conditions {
                              let g = file.group(name).unwrap();
                              let x = g.dataset("x").unwrap().read_2d::<Float>().unwrap();
                              let y = g.dataset("y").unwrap().read_2d::<Float>().unwrap();
                              assert_eq!(x.shape(), y.shape());
                              (x, y)
                          } else {
                              panic!(
                                  "Grid {} requires a valid file for setting initial size",
                                  name
                              );
                          }
                      }
                      _ => todo!(),
                      */
                };
                sbp::grid::Grid::new(x, y).unwrap()
            })
            .collect();
        let op = self
            .grids
            .iter()
            .map(|(name, g)| {
                let default_operators = default.operators.unwrap_or_default();
                let operators = g.operators.unwrap_or_default();
                let xi = operators.xi.unwrap_or_else(|| {
                    default_operators
                        .xi
                        .unwrap_or_else(|| panic!("No xi operator found for grid: {name}"))
                });
                let eta = operators.eta.unwrap_or_else(|| {
                    default_operators
                        .eta
                        .unwrap_or_else(|| panic!("No eta operator found for grid: {name}"))
                });

                use input::Operator as op;
                use sbp::operators::*;

                let matcher = |op| -> Box<dyn SbpOperator2d> {
                    match op {
                        op::Upwind4 => Box::new(Upwind4),
                        op::Upwind4h2 => Box::new(Upwind4h2),
                        op::Upwind9 => Box::new(Upwind9),
                        op::Upwind9h2 => Box::new(Upwind9h2),
                        op::Sbp4 => Box::new(SBP4),
                        op::Sbp8 => Box::new(SBP8),
                    }
                };

                Box::new((matcher(eta), matcher(xi))) as Box<dyn SbpOperator2d>
            })
            .collect();
        let boundary_conditions = self
            .grids
            .iter()
            .map(|(name, g)| {
                let default_bc = default.boundary_conditions.clone().unwrap_or_default();
                use input::BoundaryType;
                g.boundary_conditions
                    .clone()
                    .unwrap_or_default()
                    .zip(default_bc)
                    .map(|(bc, fallback)| bc.or(fallback))
                    .map(|bc| match bc {
                        None => match &boundary_conditions {
                            BoundaryConditions::Vortex(vortex) => {
                                euler::BoundaryCharacteristic::Vortex(vortex.clone())
                            }
                            BoundaryConditions::Expressions(expr) => {
                                euler::BoundaryCharacteristic::Eval(expr.clone() )
                            }
                            BoundaryConditions::NotNeeded => panic!(
                                "Boundary conditions are not available, but needed for grid {name}"
                            ),
                        },
                        Some(BoundaryType::This) => euler::BoundaryCharacteristic::This,
                        Some(BoundaryType::Vortex) => euler::BoundaryCharacteristic::Vortex(
                            if let BoundaryConditions::Vortex(vortex) = &boundary_conditions {
                                vortex.clone()
                            } else {
                                panic!("Wanted vortex boundary conditions not found, needed for grid {name}")
                            },
                        ),
                        Some(BoundaryType::Neighbour(name)) => {
                            let j = self.grids.get_index_of(&name).unwrap();
                            euler::BoundaryCharacteristic::Grid(j)
                        }
                        Some(BoundaryType::Interpolate(inp)) => {
                            let j = self.grids.get_index_of(&inp.neighbour).unwrap();
                            euler::BoundaryCharacteristic::Interpolate(
                                j,
                                inp.operator.unwrap().into(),
                            )
                        }
                        Some(BoundaryType::Multi(multi)) => {
                            euler::BoundaryCharacteristic::MultiGrid(
                                multi
                                    .iter()
                                    .map(|m| {
                                        let ineighbour =
                                            self.grids.get_index_of(&m.neighbour).unwrap();
                                        (ineighbour, m.start, m.end)
                                    })
                                    .collect(),
                            )
                        }
                    })
            })
            .collect();
        RuntimeConfiguration {
            names,
            grids,
            boundary_conditions,
            op,
            integration_time: self.integration_time,
            initial_conditions,
        }
    }
}
