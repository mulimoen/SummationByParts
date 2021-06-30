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
    /*
    #[serde(rename = "initial_conditions")]
    InitialConditions,
    */
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ArrayForm {
    /// Only know the one dimension, will broadcast to
    /// two dimensions once we know about both dims
    Array1(ndarray::Array1<Float>),
    /// The usize is the inner dimension (nx)
    Array2(ndarray::Array2<Float>),
    /*
    /// A still unknown array, will be filled out by later
    /// pass when initial_conditions file is known
    Unknown,
    */
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

impl From<GridLike> for ArrayForm {
    fn from(t: GridLike) -> Self {
        match t {
            GridLike::Linspace(lin) => Self::Array1(if lin.h2 {
                h2linspace(lin.start, lin.end, lin.steps)
            } else {
                ndarray::Array::linspace(lin.start, lin.end, lin.steps)
            }),
            GridLike::Array(arr) => arr,
            // GridLike::InitialConditions => Self::Unknown,
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
    pub operator: Option<InterpolationOperator>,
    #[serde(alias = "neighbor")]
    pub neighbour: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Multi {
    #[serde(alias = "neighbor")]
    pub neighbour: String,
    pub start: usize,
    pub end: usize,
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
/// Will be evaluated by evalexpr
pub struct ExpressionsConservation {
    pub globals: Option<String>,
    pub rho: String,
    pub rhou: String,
    pub rhov: String,
    pub e: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Will be evaluated by evalexpr
pub struct ExpressionsPressure {
    pub globals: Option<String>,
    pub rho: String,
    pub u: String,
    pub v: String,
    pub p: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[serde(untagged)]
pub enum Expressions {
    Conservation(ExpressionsConservation),
    Pressure(ExpressionsPressure),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InitialConditions {
    Vortex(euler::VortexParameters),
    // File(String),
    Expressions(Expressions),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BoundaryConditions {
    /// Initial conditions also contain the bc
    #[serde(rename = "initial_conditions")]
    InputInitialConditions,
    Vortex(euler::VortexParameters),
    Expressions(Expressions),
    #[serde(rename = "not_needed")]
    NotNeeded,
}

impl Default for BoundaryConditions {
    fn default() -> Self {
        Self::NotNeeded
    }
}

fn default_gamma() -> Float {
    1.4
}

#[derive(Clone, Debug, Serialize, Deserialize)]
/// Input configuration (json)
pub struct Configuration {
    pub grids: Grids,
    pub integration_time: Float,
    pub initial_conditions: InitialConditions,
    #[serde(default)]
    pub boundary_conditions: BoundaryConditions,
    #[serde(default = "default_gamma")]
    pub gamma: Float,
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
        initial_conditions: InitialConditions::Vortex(euler::VortexParameters {
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
        }),
        boundary_conditions: BoundaryConditions::default(),
        gamma: 1.4,
    };
    println!("{}", json5::to_string(&configuration).unwrap());
}
