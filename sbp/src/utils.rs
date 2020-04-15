use crate::grid::Grid;
use crate::Float;
use json::JsonValue;

pub struct Direction<T> {
    pub north: T,
    pub south: T,
    pub west: T,
    pub east: T,
}

#[derive(Debug, Clone)]
pub struct ExtendedGrid {
    pub grid: Grid,
    pub name: Option<String>,
    pub dire: Option<String>,
    pub dirw: Option<String>,
    pub dirn: Option<String>,
    pub dirs: Option<String>,
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
pub fn json_to_grids(json: JsonValue) -> Result<Vec<ExtendedGrid>, String> {
    fn json_to_grid(mut grid: JsonValue) -> Result<ExtendedGrid, String> {
        #[derive(Debug)]
        enum ArrayForm {
            /// Only know the one dimension, will broadcast to
            /// two dimensions once we know about both dims
            Array1(ndarray::Array1<Float>),
            /// The usize is the inner dimension (nx)
            Array2(ndarray::Array2<Float>),
        }
        if grid.is_empty() {
            return Err("empty object".to_string());
        }
        let name = grid.remove("name").take_string();
        let dire = grid.remove("dirE").take_string();
        let dirw = grid.remove("dirW").take_string();
        let dirn = grid.remove("dirN").take_string();
        let dirs = grid.remove("dirS").take_string();

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
                            Ok(x.as_number().ok_or_else(|| "Array contained something that could not be converted to an array".to_string())?.into())
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

        let x = grid.remove("x");
        if x.is_empty() {
            return Err("x was empty".to_string());
        }
        let x = to_array_form(x)?;

        let y = grid.remove("y");
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

        if !grid.is_empty() {
            eprintln!("Grid contains some unused entries");
            for i in grid.entries() {
                eprintln!("{:#?}", i);
            }
        }

        Ok(ExtendedGrid {
            grid: Grid::new(x, y).unwrap(),
            name,
            dire,
            dirw,
            dirn,
            dirs,
        })
    }

    match json {
        JsonValue::Array(a) => a
            .into_iter()
            .map(json_to_grid)
            .collect::<Result<Vec<_>, _>>(),
        grid => Ok(vec![json_to_grid(grid)?]),
    }
}

#[test]
fn parse_linspace() {
    let grids = json_to_grids(
        json::parse(r#"[{"name": "main", "x": "linspace:0:10:20", "y": "linspace:0:10:21"}]"#)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(grids.len(), 1);
    assert_eq!(grids[0].grid.x.shape(), [21, 20]);
    assert_eq!(grids[0].grid.y.shape(), [21, 20]);
    assert_eq!(grids[0].name.as_ref().unwrap(), "main");
    let grids = json_to_grids(
        json::parse(r#"{"name": "main", "x": "linspace:0:10:20", "y": "linspace:0:10:21"}"#)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(grids.len(), 1);
    assert_eq!(grids[0].grid.x.shape(), [21, 20]);
    assert_eq!(grids[0].grid.y.shape(), [21, 20]);
    assert_eq!(grids[0].name.as_ref().unwrap(), "main");
}

#[test]
fn parse_1d() {
    let grids =
        json_to_grids(json::parse(r#"{"x": [1, 2, 3, 4, 5.1, 3], "y": [1, 2]}"#).unwrap()).unwrap();
    assert_eq!(grids.len(), 1);
    let grid = &grids[0];
    assert_eq!(grid.grid.x.shape(), &[2, 6]);
    assert_eq!(grid.grid.x.shape(), grid.grid.y.shape());
}

#[test]
fn parse_2d() {
    let grids =
        json_to_grids(json::parse(r#"{"x": [[1, 2], [3, 4], [5.1, 3]], "y": [1, 2, 3]}"#).unwrap())
            .unwrap();
    assert_eq!(grids.len(), 1);
    let grid = &grids[0];
    assert_eq!(grid.grid.x.shape(), &[3, 2]);
    assert_eq!(grid.grid.x.shape(), grid.grid.y.shape());
    json_to_grids(
        json::parse(r#"{"x": [[1, 2], [3, 4], [5.1, 3], [1]], "y": [1, 2, 3]}"#).unwrap(),
    )
    .unwrap_err();
    json_to_grids(
        json::parse(r#"{"y": [[1, 2], [3, 4], [5.1, 3], [1]], "x": [1, 2, 3]}"#).unwrap(),
    )
    .unwrap_err();
    let grids = json_to_grids(
        json::parse(r#"{"x": [[1, 2], [3, 4], [5.1, 3]], "y": [[1, 2], [3, 4], [5, 6]]}"#).unwrap(),
    )
    .unwrap();
    assert_eq!(grids.len(), 1);
    json_to_grids(
        json::parse(r#"{"x": [[1, 2], [3, 4], [5.1, 3]], "y": [[1, 2], [3, 4], [5]]}"#).unwrap(),
    )
    .unwrap_err();
}

#[test]
fn parse_err() {
    json_to_grids(json::parse(r#"{}"#).unwrap()).unwrap_err();
    json_to_grids(json::parse(r#"0.45"#).unwrap()).unwrap_err();
    json_to_grids(json::parse(r#"{"x": "linspace", "y": [0.1, 0.2]}"#).unwrap()).unwrap_err();
    json_to_grids(json::parse(r#"{"x": "linspace:::", "y": [0.1, 0.2]}"#).unwrap()).unwrap_err();
    json_to_grids(json::parse(r#"{"x": "linspace:1.2:3.1:412.2", "y": [0.1, 0.2]}"#).unwrap())
        .unwrap_err();
    json_to_grids(json::parse(r#"{"x": [-2, -3, "dfd"], "y": [0.1, 0.2]}"#).unwrap()).unwrap_err();
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

pub fn h2linspace(start: Float, end: Float, n: usize) -> ndarray::Array1<Float> {
    let h = (end - start) / (n - 2) as Float;
    ndarray::Array1::from_shape_fn(n, |i| match i {
        0 => start,
        i if i == n - 1 => end,
        i => start + h * (i as Float - 0.5),
    })
}

#[test]
fn test_h2linspace() {
    let x = h2linspace(-1.0, 1.0, 50);
    println!("{}", x);
    approx::assert_abs_diff_eq!(x[0], -1.0, epsilon = 1e-6);
    approx::assert_abs_diff_eq!(x[49], 1.0, epsilon = 1e-6);
    let hend = x[1] - x[0];
    let h = x[2] - x[1];
    approx::assert_abs_diff_eq!(x[49] - x[48], hend, epsilon = 1e-6);
    approx::assert_abs_diff_eq!(2.0 * hend, h, epsilon = 1e-6);
    for i in 1..48 {
        approx::assert_abs_diff_eq!(x[i + 1] - x[i], h, epsilon = 1e-6);
    }
}
