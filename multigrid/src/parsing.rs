use super::DiffOp;
use either::*;
use sbp::utils::h2linspace;
use sbp::Float;

use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Operator {
    Upwind4,
    Upwind9,
    Upwind4h2,
    Upwind9h2,
    Sbp4,
    Sbp8,
}

#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct Operators {
    pub xi: Option<Operator>,
    pub eta: Option<Operator>,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct Linspace {
    pub start: Float,
    pub end: Float,
    pub steps: usize,
    #[serde(default)]
    pub h2: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GridLike {
    Linspace(Linspace),
    Array(ArrayForm),
}

impl From<GridLike> for ArrayForm {
    fn from(t: GridLike) -> Self {
        match t {
            GridLike::Linspace(lin) => Self::Array1(if lin.h2 {
                h2linspace(lin.start, lin.end, lin.steps)
            } else {
                ndarray::Array::linspace(lin.start, lin.end, lin.steps)
            }),
            GridLike::Array(arr) => arr,
        }
    }
}

impl From<Linspace> for GridLike {
    fn from(t: Linspace) -> Self {
        Self::Linspace(t)
    }
}

impl From<ArrayForm> for GridLike {
    fn from(t: ArrayForm) -> Self {
        Self::Array(t)
    }
}

impl From<ndarray::Array1<Float>> for GridLike {
    fn from(t: ndarray::Array1<Float>) -> Self {
        Self::Array(t.into())
    }
}

impl From<ndarray::Array2<Float>> for GridLike {
    fn from(t: ndarray::Array2<Float>) -> Self {
        Self::Array(t.into())
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum InterpolationOperator {
    #[serde(rename = "4")]
    Four,
    #[serde(rename = "8")]
    Eight,
    #[serde(rename = "9")]
    Nine,
    #[serde(rename = "9h2")]
    NineH2,
}

impl Into<Box<dyn sbp::operators::InterpolationOperator>> for InterpolationOperator {
    fn into(self) -> Box<dyn sbp::operators::InterpolationOperator> {
        use sbp::operators::{Interpolation4, Interpolation8, Interpolation9, Interpolation9h2};
        match self {
            InterpolationOperator::Four => Box::new(Interpolation4),
            InterpolationOperator::Eight => Box::new(Interpolation8),
            InterpolationOperator::Nine => Box::new(Interpolation9),
            InterpolationOperator::NineH2 => Box::new(Interpolation9h2),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Interpolate {
    operator: Option<InterpolationOperator>,
    #[serde(alias = "neighbor")]
    neighbour: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Multi {
    #[serde(alias = "neighbor")]
    neighbour: String,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BoundaryType {
    This,
    Interpolate(Interpolate),
    #[serde(alias = "neighbor")]
    Neighbour(String),
    Vortex,
    Multi(Vec<Multi>),
}

pub type BoundaryDescriptors = sbp::utils::Direction<Option<BoundaryType>>;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct GridConfig {
    pub operators: Option<Operators>,
    pub x: Option<GridLike>,
    pub y: Option<GridLike>,
    pub boundary_conditions: Option<BoundaryDescriptors>,
}

type Grids = indexmap::IndexMap<String, GridConfig>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Configuration {
    pub grids: Grids,
    pub integration_time: Float,
    pub vortex: euler::VortexParameters,
}

pub struct RuntimeConfiguration {
    pub names: Vec<String>,
    pub grids: Vec<sbp::grid::Grid>,
    pub bc: Vec<euler::BoundaryCharacteristics>,
    pub op: Vec<DiffOp>,
    pub integration_time: Float,
    pub vortex: euler::VortexParameters,
}

impl Configuration {
    pub fn to_runtime(mut self) -> RuntimeConfiguration {
        let default = self.grids.shift_remove("default").unwrap_or_default();
        let names = self.grids.keys().cloned().collect();
        let grids = self
            .grids
            .iter()
            .map(|(_name, g)| {
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
                    }
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
                        .unwrap_or_else(|| panic!("No xi operator found for grid: {}", name))
                });
                let eta = operators.eta.unwrap_or_else(|| {
                    default_operators
                        .eta
                        .unwrap_or_else(|| panic!("No eta operator found for grid: {}", name))
                });

                use sbp::operators::*;
                use Operator as op;
                match (eta, xi) {
                    (op::Upwind4, op::Upwind4) => {
                        Right(Box::new(Upwind4) as Box<dyn UpwindOperator2d>)
                    }
                    (op::Upwind4h2, op::Upwind4h2) => {
                        Right(Box::new(Upwind4h2) as Box<dyn UpwindOperator2d>)
                    }
                    (op::Upwind9, op::Upwind9) => {
                        Right(Box::new(Upwind9) as Box<dyn UpwindOperator2d>)
                    }
                    (op::Upwind9h2, op::Upwind9h2) => {
                        Right(Box::new(Upwind9h2) as Box<dyn UpwindOperator2d>)
                    }
                    (op::Upwind4, op::Upwind4h2) => {
                        Right(Box::new((&Upwind4, &Upwind4h2)) as Box<dyn UpwindOperator2d>)
                    }
                    (op::Upwind9, op::Upwind9h2) => {
                        Right(Box::new((&Upwind9, &Upwind9h2)) as Box<dyn UpwindOperator2d>)
                    }
                    (op::Upwind9h2, op::Upwind9) => {
                        Right(Box::new((&Upwind9h2, &Upwind9)) as Box<dyn UpwindOperator2d>)
                    }
                    (op::Sbp4, op::Sbp4) => Left(Box::new(SBP4) as Box<dyn SbpOperator2d>),
                    (op::Sbp8, op::Sbp8) => Left(Box::new(SBP8) as Box<dyn SbpOperator2d>),
                    _ => todo!("Combination {:?}, {:?} not implemented", eta, xi),
                }
            })
            .collect();
        let bc = self
            .grids
            .iter()
            .enumerate()
            .map(|(i, (_name, g))| {
                let default_bc = default.boundary_conditions.clone().unwrap_or_default();
                g.boundary_conditions
                    .clone()
                    .unwrap_or_default()
                    .zip(default_bc)
                    .map(|(bc, fallback)| bc.or(fallback))
                    .map(|bc| match bc {
                        None | Some(BoundaryType::Vortex) => {
                            euler::BoundaryCharacteristic::Vortex(self.vortex.clone())
                        }
                        Some(BoundaryType::This) => euler::BoundaryCharacteristic::Grid(i),
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
            bc,
            op,
            integration_time: self.integration_time,
            vortex: self.vortex,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ArrayForm {
    /// Only know the one dimension, will broadcast to
    /// two dimensions once we know about both dims
    Array1(ndarray::Array1<Float>),
    /// The usize is the inner dimension (nx)
    Array2(ndarray::Array2<Float>),
}

impl From<ndarray::Array1<Float>> for ArrayForm {
    fn from(t: ndarray::Array1<Float>) -> Self {
        Self::Array1(t)
    }
}

impl From<ndarray::Array2<Float>> for ArrayForm {
    fn from(t: ndarray::Array2<Float>) -> Self {
        Self::Array2(t)
    }
}

#[test]
fn output_configuration() {
    let mut grids = Grids::new();
    grids.insert(
        "default".to_string(),
        GridConfig {
            boundary_conditions: None,
            x: None,
            y: None,
            operators: None,
        },
    );
    grids.insert(
        "operators1".to_string(),
        GridConfig {
            boundary_conditions: None,
            x: None,
            y: None,
            operators: Some(Operators {
                xi: Some(Operator::Upwind4),
                eta: Some(Operator::Upwind9),
            }),
        },
    );
    grids.insert(
        "operators2".to_string(),
        GridConfig {
            boundary_conditions: None,
            x: None,
            y: None,
            operators: Some(Operators {
                xi: Some(Operator::Upwind4h2),
                eta: Some(Operator::Upwind9h2),
            }),
        },
    );
    grids.insert(
        "operators3".to_string(),
        GridConfig {
            boundary_conditions: None,
            x: None,
            y: None,
            operators: Some(Operators {
                xi: Some(Operator::Sbp4),
                eta: Some(Operator::Sbp8),
            }),
        },
    );
    grids.insert(
        "linspaced".to_string(),
        GridConfig {
            boundary_conditions: None,
            x: Some(
                Linspace {
                    start: 0.0,
                    end: 1.0,
                    steps: 32,
                    h2: false,
                }
                .into(),
            ),
            y: Some(
                Linspace {
                    start: -1.0,
                    end: 1.0,
                    steps: 35,
                    h2: true,
                }
                .into(),
            ),
            operators: None,
        },
    );
    grids.insert(
        "array1".to_string(),
        GridConfig {
            boundary_conditions: None,
            x: Some(ndarray::arr1(&[1.0, 2.0, 3.0, 4.0]).into()),
            y: Some(ndarray::arr1(&[-4.0, -3.0, -2.0, -1.0, 0.0]).into()),
            operators: None,
        },
    );
    grids.insert(
        "array2".to_string(),
        GridConfig {
            boundary_conditions: None,
            x: Some(ndarray::arr2(&[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]).into()),
            y: Some(ndarray::arr2(&[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]).into()),
            operators: None,
        },
    );
    grids.insert(
        "boundary_conditions".to_string(),
        GridConfig {
            boundary_conditions: Some(BoundaryDescriptors {
                north: None,
                south: Some(BoundaryType::This),
                east: Some(BoundaryType::Neighbour("name_of_grid".to_string())),
                west: Some(BoundaryType::Vortex),
            }),
            x: None,
            y: None,
            operators: None,
        },
    );
    grids.insert(
        "boundary_conditions_interpolation".to_string(),
        GridConfig {
            boundary_conditions: Some(BoundaryDescriptors {
                north: Some(BoundaryType::Interpolate(Interpolate {
                    neighbour: "name_of_grid".to_string(),
                    operator: Some(InterpolationOperator::Four),
                })),
                south: Some(BoundaryType::Interpolate(Interpolate {
                    neighbour: "name_of_grid".to_string(),
                    operator: Some(InterpolationOperator::Nine),
                })),
                west: Some(BoundaryType::Interpolate(Interpolate {
                    neighbour: "name_of_grid".to_string(),
                    operator: Some(InterpolationOperator::Eight),
                })),
                east: Some(BoundaryType::Interpolate(Interpolate {
                    neighbour: "name_of_grid".to_string(),
                    operator: Some(InterpolationOperator::NineH2),
                })),
            }),
            x: None,
            y: None,
            operators: None,
        },
    );
    grids.insert(
        "boundary_conditions_multigrid".to_string(),
        GridConfig {
            boundary_conditions: Some(BoundaryDescriptors {
                north: Some(BoundaryType::Multi(vec![Multi {
                    neighbour: "name_of_grid".to_string(),
                    start: 4,
                    end: 7,
                }])),
                south: Some(BoundaryType::Multi(vec![
                    Multi {
                        neighbour: "name_of_grid".to_string(),
                        start: 4,
                        end: 7,
                    },
                    Multi {
                        neighbour: "name_of_grid".to_string(),
                        start: 41,
                        end: 912,
                    },
                ])),
                east: None,
                west: None,
            }),
            x: None,
            y: None,
            operators: None,
        },
    );
    let configuration = Configuration {
        grids,
        integration_time: 2.0,
        vortex: euler::VortexParameters {
            mach: 0.5,
            vortices: {
                let mut arr = euler::ArrayVec::new();
                arr.push(euler::Vortice {
                    eps: 1.0,
                    x0: -1.0,
                    y0: 0.0,
                    rstar: 0.5,
                });
                arr
            },
        },
    };
    println!("{}", json5::to_string(&configuration).unwrap());
}
