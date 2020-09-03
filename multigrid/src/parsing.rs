use super::DiffOp;
use either::*;
use json::JsonValue;
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
#[serde(rename_all = "lowercase")]
pub enum BoundaryType {
    This,
    Interpolate(Interpolate),
    #[serde(alias = "neighbor")]
    Neighbour(String),
    Vortex,
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
                let xi = operators.xi.unwrap_or(
                    default_operators
                        .xi
                        .unwrap_or_else(|| panic!("No xi operator found for grid: {}", name)),
                );
                let eta = operators.eta.unwrap_or(
                    default_operators
                        .eta
                        .unwrap_or_else(|| panic!("No eta operator found for grid: {}", name)),
                );

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
                    (op::Sbp4, op::Sbp4) => Left(Box::new(SBP4) as Box<dyn SbpOperator2d>),
                    (op::Sbp8, op::Sbp8) => Left(Box::new(SBP8) as Box<dyn SbpOperator2d>),
                    _ => todo!(),
                }
            })
            .collect();
        let bc = self
            .grids
            .iter()
            .enumerate()
            .map(|(i, (_name, g))| {
                g.boundary_conditions
                    .clone()
                    .unwrap_or_else(|| default.boundary_conditions.clone().unwrap_or_default())
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
        };
        todo!();
    }
}

pub fn json_to_grids(
    mut jsongrids: JsonValue,
    vortexparams: euler::VortexParameters,
) -> (
    Vec<String>,
    Vec<sbp::grid::Grid>,
    Vec<euler::BoundaryCharacteristics>,
    Vec<DiffOp>,
) {
    let default = jsongrids.remove("default");
    let default_operator = {
        let operators = &default["operators"];
        let defaultxi = operators["xi"].as_str().unwrap_or("upwind4");
        let defaulteta = operators["eta"].as_str().unwrap_or("upwind4");

        (defaulteta.to_string(), defaultxi.to_string())
    };
    /*
    let default_bc: sbp::utils::Direction<Option<String>> = {
        let bc = &default["boundary_conditions"];
        Direction {
            south: bc["south"].as_str().map(|x| x.to_string()),
            north: bc["north"].as_str().map(|x| x.to_string()),
            west: bc["west"].as_str().map(|x| x.to_string()),
            east: bc["east"].as_str().map(|x| x.to_string()),
        }
    };
    */
    let mut names = Vec::new();
    let mut grids = Vec::new();

    let mut operators: Vec<DiffOp> = Vec::new();
    for (name, grid) in jsongrids.entries() {
        names.push(name.to_string());
        grids.push(json2grid(grid["x"].clone(), grid["y"].clone()).unwrap());

        operators.push({
            use sbp::operators::*;
            let opxi = grid["operators"]["xi"]
                .as_str()
                .unwrap_or(&default_operator.1);

            let opeta = grid["operators"]["eta"]
                .as_str()
                .unwrap_or(&default_operator.0);

            match (opeta, opxi) {
                ("upwind4", "upwind4") => Right(Box::new(Upwind4) as Box<dyn UpwindOperator2d>),
                ("upwind9", "upwind9") => Right(Box::new(Upwind9) as Box<dyn UpwindOperator2d>),
                ("upwind4h2", "upwind4h2") => {
                    Right(Box::new(Upwind4h2) as Box<dyn UpwindOperator2d>)
                }
                ("upwind9h2", "upwind9h2") => {
                    Right(Box::new(Upwind9h2) as Box<dyn UpwindOperator2d>)
                }

                ("upwind4", "upwind9") => {
                    Right(Box::new((&Upwind4, &Upwind9)) as Box<dyn UpwindOperator2d>)
                }
                ("upwind4", "upwind4h2") => {
                    Right(Box::new((&Upwind4, &Upwind4h2)) as Box<dyn UpwindOperator2d>)
                }
                ("upwind4", "upwind9h2") => {
                    Right(Box::new((&Upwind4, &Upwind9h2)) as Box<dyn UpwindOperator2d>)
                }

                ("upwind9", "upwind4") => {
                    Right(Box::new((&Upwind9, &Upwind4)) as Box<dyn UpwindOperator2d>)
                }
                ("upwind9", "upwind4h2") => {
                    Right(Box::new((&Upwind9, &Upwind4h2)) as Box<dyn UpwindOperator2d>)
                }
                ("upwind9", "upwind9h2") => {
                    Right(Box::new((&Upwind9, &Upwind9h2)) as Box<dyn UpwindOperator2d>)
                }

                ("upwind4h2", "upwind4") => {
                    Right(Box::new((&Upwind4h2, &Upwind4)) as Box<dyn UpwindOperator2d>)
                }
                ("upwind4h2", "upwind9") => {
                    Right(Box::new((&Upwind4h2, &Upwind9)) as Box<dyn UpwindOperator2d>)
                }
                ("upwind4h2", "upwind9h2") => {
                    Right(Box::new((&Upwind4h2, &Upwind9h2)) as Box<dyn UpwindOperator2d>)
                }

                ("upwind9h2", "upwind4") => {
                    Right(Box::new((&Upwind9h2, &Upwind4)) as Box<dyn UpwindOperator2d>)
                }
                ("upwind9h2", "upwind9") => {
                    Right(Box::new((&Upwind9h2, &Upwind9)) as Box<dyn UpwindOperator2d>)
                }
                ("upwind9h2", "upwind4h2") => {
                    Right(Box::new((&Upwind9h2, &Upwind4h2)) as Box<dyn UpwindOperator2d>)
                }

                (opeta, opxi) => panic!("combination {} {} not yet implemented", opeta, opxi),
            }
        });
    }

    let mut bcs = Vec::new();
    let determine_bc = |dir: Option<&str>| match dir {
        Some(dir) => {
            if dir == "vortex" {
                euler::BoundaryCharacteristic::Vortex(vortexparams.clone())
            } else if let Some(grid) = dir.strip_prefix("interpolate:") {
                use sbp::operators::*;
                let (grid, int_op) = if let Some(rest) = grid.strip_prefix("4:") {
                    (
                        rest,
                        Box::new(Interpolation4) as Box<dyn InterpolationOperator>,
                    )
                } else if let Some(rest) = grid.strip_prefix("9:") {
                    (
                        rest,
                        Box::new(Interpolation9) as Box<dyn InterpolationOperator>,
                    )
                } else if let Some(rest) = grid.strip_prefix("8:") {
                    (
                        rest,
                        Box::new(Interpolation8) as Box<dyn InterpolationOperator>,
                    )
                } else if let Some(rest) = grid.strip_prefix("9h2:") {
                    (
                        rest,
                        Box::new(Interpolation9h2) as Box<dyn InterpolationOperator>,
                    )
                } else {
                    (
                        grid,
                        Box::new(Interpolation4) as Box<dyn InterpolationOperator>,
                    )
                };
                euler::BoundaryCharacteristic::Interpolate(
                    names.iter().position(|other| other == grid).unwrap(),
                    int_op,
                )
            } else if let Some(multigrid) = dir.strip_prefix("multi:") {
                let grids = multigrid.split(':');
                euler::BoundaryCharacteristic::MultiGrid(
                    grids
                        .map(|g| {
                            let rparen = g.find('(').unwrap();
                            let gridname = &g[..rparen];

                            let gridnumber =
                                names.iter().position(|other| other == gridname).unwrap();

                            let paren = &g[rparen + 1..];
                            let paren = &paren[..paren.len() - 1];
                            let mut pareni = paren.split(',');
                            let start = pareni.next().unwrap().parse::<usize>().unwrap();
                            let end = pareni.next().unwrap().parse::<usize>().unwrap();

                            (gridnumber, start, end)
                        })
                        .collect::<Vec<_>>(),
                )
            } else {
                euler::BoundaryCharacteristic::Grid(
                    names.iter().position(|other| other == dir).unwrap(),
                )
            }
        }
        None => euler::BoundaryCharacteristic::This,
    };
    for name in &names {
        let bc = &jsongrids[name]["boundary_conditions"];
        let bc_n = determine_bc(bc["north"].as_str());
        let bc_s = determine_bc(bc["south"].as_str());
        let bc_e = determine_bc(bc["east"].as_str());
        let bc_w = determine_bc(bc["west"].as_str());

        let bc = euler::BoundaryCharacteristics {
            north: bc_n,
            south: bc_s,
            east: bc_e,
            west: bc_w,
        };

        bcs.push(bc);
    }

    (names, grids, bcs, operators)
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

/// Parsing json strings to some gridlike form
///
/// Each grid should be an object with the descriptors on the form
///
/// x: [x0, x1, ..., xn]
/// which results in x being broadcasted to nx/ny size
/// x: linspace:start:end:num
/// x: linspace:h2:start:end:num
/// where x will be from start to end inclusive, with num steps
/// x: [[x00, x01, .., x0n], [x10, x11, .., x1n], ... [xm0, xm1, ..., xmn]]
/// which is the full grid x
///
/// This conversion is similar with y
///
/// Optional parameters:
/// * name (for relating boundaries)
/// * dir{e,w,n,s} (for boundary terms)
fn json2grid(x: JsonValue, y: JsonValue) -> Result<sbp::grid::Grid, String> {
    let to_array_form = |mut x: JsonValue| {
        if let Some(s) = x.take_string() {
            if let Some(s) = s.strip_prefix("linspace:") {
                let (s, h2) = if let Some(s) = s.strip_prefix("h2:") {
                    (s, true)
                } else {
                    (s, false)
                };

                // linspace:start:stop:steps
                let mut iter = s.split(':');

                let start = iter.next();
                let start: Float = match start {
                    Some(x) => x.parse().map_err(|e| format!("linspace: {}", e))?,
                    None => return Err(format!("")),
                };
                let end = iter.next();
                let end: Float = match end {
                    Some(x) => x.parse().map_err(|e| format!("linspace: {}", e))?,
                    None => return Err(format!("")),
                };
                let steps = iter.next();
                let steps: usize = match steps {
                    Some(x) => x.parse().map_err(|e| format!("linspace: {}", e))?,
                    None => return Err(format!("")),
                };
                if iter.next().is_some() {
                    return Err("linspace: contained more than expected".to_string());
                }
                Ok(ArrayForm::Array1(if h2 {
                    h2linspace(start, end, steps)
                } else {
                    ndarray::Array::linspace(start, end, steps)
                }))
            } else {
                Err("Could not parse gridline".to_string())
            }
        } else if x.is_array() {
            let arrlen = x.len();
            if arrlen == 0 {
                return Err("gridline does not have any members".to_string());
            }
            if !x[0].is_array() {
                let v = x
                    .members()
                    .map(|x: &JsonValue| -> Result<Float, String> {
                        Ok(x.as_number()
                            .ok_or_else(|| {
                                "Array contained something that could not be converted to an array"
                                    .to_string()
                            })?
                            .into())
                    })
                    .collect::<Result<Vec<Float>, _>>()?;
                Ok(ArrayForm::Array1(ndarray::Array::from(v)))
            } else {
                let arrlen2 = x[0].len();
                if arrlen2 == 0 {
                    return Err("gridline does not have any members".to_string());
                }
                for member in x.members() {
                    if arrlen2 != member.len() {
                        return Err("some arrays seems to have differing lengths".to_string());
                    }
                }
                let mut arr = ndarray::Array::zeros((arrlen, arrlen2));
                for (mut arr, member) in arr.outer_iter_mut().zip(x.members()) {
                    for (a, m) in arr.iter_mut().zip(member.members()) {
                        *a = m
                            .as_number()
                            .ok_or_else(|| {
                                "array contained something which was not a number".to_string()
                            })?
                            .into()
                    }
                }
                Ok(ArrayForm::Array2(arr))
            }
        } else {
            Err("Inner object was not a string value, or an array".to_string())
        }
    };

    if x.is_empty() {
        return Err("x was empty".to_string());
    }
    let x = to_array_form(x)?;

    if y.is_empty() {
        return Err("y was empty".to_string());
    }
    let y = to_array_form(y)?;

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
        (ArrayForm::Array2(x), ArrayForm::Array2(y)) => {
            assert_eq!(x.shape(), y.shape());
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
    };
    assert_eq!(x.shape(), y.shape());

    Ok(sbp::grid::Grid::new(x, y).unwrap())
}

pub fn json_to_vortex(mut json: JsonValue) -> euler::VortexParameters {
    let mach = json.remove("mach").as_number().unwrap().into();

    // Get max length of any (potential) array
    let mut maxlen = None;
    for &name in &["x0", "y0", "rstar", "eps"] {
        if json[name].is_array() {
            maxlen = Some(json[name].members().count());
            break;
        }
    }
    let maxlen = maxlen.unwrap_or(1);

    let into_iterator = move |elem| -> Box<dyn Iterator<Item = Float>> {
        match elem {
            JsonValue::Number(x) => Box::new(std::iter::repeat(x.into())),
            JsonValue::Array(x) => {
                Box::new(x.into_iter().map(move |x| x.as_number().unwrap().into()))
            }
            _ => panic!("This element is not a number of array"),
        }
    };

    let x0 = into_iterator(json.remove("x0"));
    let y0 = into_iterator(json.remove("y0"));
    let rstar = into_iterator(json.remove("rstar"));
    let eps = into_iterator(json.remove("eps"));

    let mut vortices = euler::ArrayVec::new();
    for (((x0, y0), rstar), eps) in x0.zip(y0).zip(rstar).zip(eps).take(maxlen) {
        vortices.push(euler::Vortice { x0, y0, rstar, eps })
    }

    if !json.is_empty() {
        eprintln!("Found unused items when parsing vortex");
        for (name, val) in json.entries() {
            eprintln!("\t{} {}", name, val.dump());
        }
    }

    euler::VortexParameters { vortices, mach }
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
