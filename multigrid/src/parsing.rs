use super::DiffOp;
use crate::grid::Grid;
use crate::Float;
use either::*;
use json::JsonValue;
use sbp::utils::h2linspace;

pub fn json_to_grids(
    mut jsongrids: JsonValue,
    vortexparams: sbp::euler::VortexParameters,
) -> (
    Vec<String>,
    Vec<sbp::grid::Grid>,
    Vec<sbp::euler::BoundaryCharacteristics>,
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
                .unwrap_or(&default_operator.1);

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
                sbp::euler::BoundaryCharacteristic::Vortex(vortexparams)
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
                sbp::euler::BoundaryCharacteristic::Interpolate(
                    names.iter().position(|other| other == grid).unwrap(),
                    int_op,
                )
            } else {
                sbp::euler::BoundaryCharacteristic::Grid(
                    names.iter().position(|other| other == dir).unwrap(),
                )
            }
        }
        None => sbp::euler::BoundaryCharacteristic::This,
    };
    for name in &names {
        let bc = &jsongrids[name]["boundary_conditions"];
        let bc_n = determine_bc(bc["north"].as_str());
        let bc_s = determine_bc(bc["south"].as_str());
        let bc_e = determine_bc(bc["east"].as_str());
        let bc_w = determine_bc(bc["west"].as_str());

        let bc = sbp::euler::BoundaryCharacteristics {
            north: bc_n,
            south: bc_s,
            east: bc_e,
            west: bc_w,
        };

        bcs.push(bc);
    }

    (names, grids, bcs, operators)
}
#[derive(Debug)]
enum ArrayForm {
    /// Only know the one dimension, will broadcast to
    /// two dimensions once we know about both dims
    Array1(ndarray::Array1<Float>),
    /// The usize is the inner dimension (nx)
    Array2(ndarray::Array2<Float>),
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
fn json2grid(x: JsonValue, y: JsonValue) -> Result<Grid, String> {
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

    Ok(Grid::new(x, y).unwrap())
}

pub fn json_to_vortex(mut json: JsonValue) -> super::euler::VortexParameters {
    let x0 = json.remove("x0").as_number().unwrap().into();
    let y0 = json.remove("y0").as_number().unwrap().into();
    let mach = json.remove("mach").as_number().unwrap().into();
    let rstar = json.remove("rstar").as_number().unwrap().into();
    let eps = json.remove("eps").as_number().unwrap().into();

    if !json.is_empty() {
        eprintln!("Found unused items when parsing vortex");
        for (name, val) in json.entries() {
            eprintln!("\t{} {}", name, val.dump());
        }
    }

    super::euler::VortexParameters {
        x0,
        y0,
        mach,
        rstar,
        eps,
    }
}
