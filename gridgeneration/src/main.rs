#![allow(unused)]
use gridgeneration::*;
use plotters::prelude::*;

fn main() {
    // let grid = Grid::unit(10, 11);
    // let grid = shell(30, 21, 0.1, 3.1);
    let grid = {
        let ybot = (0..100)
            .map(|x| (x as Float / 30.0).sin())
            .collect::<Vec<_>>();
        let ytop = (0..100)
            .map(|x| 2.5 + (x as Float / 10.0).sin())
            .collect::<Vec<_>>();
        // vertical_interpolation(10, &ytop, &ybot)
        // horizontal_interpolation(10, &ytop, &ybot)
        /*
        horizontal_interpolation_dual(
            30,
            (&[0.0, 0.0, 1.0, 3.0], &[0.0, 2.0, 4.0, 6.0]),
            (&[3.0, 5.0, 6.0, 7.0], &[0.0, 2.0, 3.0, 10.0]),
        )
        */
        /*
        horizontal_interpolation_dual2(
            30,
            40,
            move |t| ((t * 3.0).sin(), t + 0.2),
            move |t| ((5.0 * t).sin() + 2.5, t),
        )
        */
        horizontal_interpolation_dual2(
            30,
            40,
            move |t| (2.0 * t * std::f64::consts::PI).sin_cos(),
            move |t| {
                let (y, x) = (2.0 * t * std::f64::consts::PI).sin_cos();
                (0.1 * y, 0.1 * x)
            },
        )
    };
    draw_grid(&grid, "grid.svg");

    let grid = horizontal_interpolation_dual2(
        200,
        200,
        move |t| {
            let ts = [0.05, 0.25, 0.75, 0.95];
            if t < ts[0] {
                let t = t / (ts[0] - 0.0);
                (t, 5.0)
            } else if t < ts[1] {
                let t = (t - ts[0]) / (ts[1] - ts[0]);
                (1.0, 5.0 - 5.0 * t)
            } else if t < ts[2] {
                let t = (t - ts[1]) / (ts[2] - ts[1]);
                (std::f64::consts::PI * (t + 0.5)).sin_cos()
            } else if t < ts[3] {
                let t = (t - ts[2]) / (ts[3] - ts[2]);
                (-1.0, 5.0 * t)
            } else {
                let t = (t - ts[3]) / (1.0 - ts[3]);
                (-1.0 + t, 5.0)
            }
        },
        move |t| {
            let (y, x) = (2.0 * t * std::f64::consts::PI).sin_cos();
            (0.2 * y, 0.2 * x)
        },
    );
    draw_grid(&grid, "ctype.svg");

    #[cfg(feature = "serde")]
    println!("{}", json5::to_string(&grid).unwrap());
}

fn linspace(range: std::ops::Range<Float>, steps: usize) -> Vec<Float> {
    let start = range.start;
    let end = range.end;

    let dx = (end - start) / (steps - 1) as Float;

    (0..steps).map(|i| i as Float * dx).collect()
}

fn draw_grid(grid: &Grid, name: &str) {
    let drawing_area = SVGBackend::new(name, (1000, 1000)).into_drawing_area();
    drawing_area.fill(&WHITE).unwrap();

    let x_minmax = grid
        .x()
        .iter()
        .fold((Float::INFINITY, -Float::INFINITY), |(min, max), &x| {
            (min.min(x), max.max(x))
        });
    let y_minmax = grid
        .y()
        .iter()
        .fold((Float::INFINITY, -Float::INFINITY), |(min, max), &y| {
            (min.min(y), max.max(y))
        });

    let mut chart = ChartBuilder::on(&drawing_area)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_minmax.0..x_minmax.1, y_minmax.0..y_minmax.1)
        .unwrap();
    chart
        .configure_mesh()
        .x_desc("x")
        .y_desc("y")
        .disable_mesh()
        .draw()
        .unwrap();

    let style = &BLACK.stroke_width(1);
    for axes in &[ndarray::Axis(1), ndarray::Axis(0)] {
        for (linex, liney) in grid.x().axis_iter(*axes).zip(grid.y().axis_iter(*axes)) {
            chart
                .draw_series(LineSeries::new(
                    linex.iter().zip(&liney).map(|(&x, &y)| (x, y)),
                    style.clone(),
                ))
                .unwrap();
        }
    }
    /*
    let coord = |idx| (grid.x()[idx], grid.y()[idx]);
    let zero = coord((0, 0));
    let up = coord((1, 0));
    let right = coord((0, 1));
    chart
        .draw_series(PointSeries::of_element(
            [zero].iter().copied(),
            5,
            ShapeStyle::from(&RED).filled(),
            &|coord, size, style| {
                EmptyElement::at(coord)
                    + Circle::new((0, 0), size, style)
                    + Text::new("η", (1, 1), ("arial", 10))
            },
        ))
        .unwrap();
    */
}
