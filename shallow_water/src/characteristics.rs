#![allow(non_snake_case)]
/*
 *                     Code generated with sympy 1.7.dev
 *
 *              See http://www.sympy.org/ for more information.
 *
 *                       This file is part of 'project'
 */
use super::Float;

pub fn Aplus(eta: Float, etau: Float, etav: Float, g: Float) -> [[Float; 3]; 3] {
    [
        [
            (eta.powf(3.0 / 2.0) * g.sqrt() + etau.abs()) / (2.0 * eta),
            1.0 / 2.0,
            0.0,
        ],
        [
            eta * g / 2.0 - etau.powi(2) / (2.0 * eta.powi(2)),
            (eta.powf(3.0 / 2.0) * g.sqrt() + 2.0 * etau + etau.abs()) / (2.0 * eta),
            0.0,
        ],
        [
            -etau * etav / (2.0 * eta.powi(2)),
            etav / (2.0 * eta),
            (eta.powf(3.0 / 2.0) * g.sqrt() + etau + etau.abs()) / (2.0 * eta),
        ],
    ]
}

pub fn Aminus(eta: Float, etau: Float, etav: Float, g: Float) -> [[Float; 3]; 3] {
    [
        [
            -(eta.powf(3.0 / 2.0) * g.sqrt() + etau.abs()) / (2.0 * eta),
            1.0 / 2.0,
            0.0,
        ],
        [
            eta * g / 2.0 - etau.powi(2) / (2.0 * eta.powi(2)),
            (-eta.powf(3.0 / 2.0) * g.sqrt() + 2.0 * etau - etau.abs()) / (2.0 * eta),
            0.0,
        ],
        [
            -etau * etav / (2.0 * eta.powi(2)),
            etav / (2.0 * eta),
            (-eta.powf(3.0 / 2.0) * g.sqrt() + etau - etau.abs()) / (2.0 * eta),
        ],
    ]
}

pub fn Bplus(eta: Float, etau: Float, etav: Float, g: Float) -> [[Float; 3]; 3] {
    [
        [
            (eta.powf(3.0 / 2.0) + etav.abs()) / (2.0 * eta),
            0.0,
            1.0 / 2.0,
        ],
        [
            -etau * etav / (2.0 * eta.powi(2)),
            (eta.powf(3.0 / 2.0) * g.sqrt() + etav + etav.abs()) / (2.0 * eta),
            etau / (2.0 * eta),
        ],
        [
            eta * g / 2.0 - etav.powi(2) / (2.0 * eta.powi(2)),
            0.0,
            (eta.powf(3.0 / 2.0) * g.sqrt() + 2.0 * etav + etav.abs()) / (2.0 * eta),
        ],
    ]
}

pub fn Bminus(eta: Float, etau: Float, etav: Float, g: Float) -> [[Float; 3]; 3] {
    [
        [
            -(eta.powf(3.0 / 2.0) + etav.abs()) / (2.0 * eta),
            0.0,
            1.0 / 2.0,
        ],
        [
            -etau * etav / (2.0 * eta.powi(2)),
            (-eta.powf(3.0 / 2.0) * g.sqrt() + etav - etav.abs()) / (2.0 * eta),
            etau / (2.0 * eta),
        ],
        [
            eta * g / 2.0 - etav.powi(2) / (2.0 * eta.powi(2)),
            0.0,
            (-eta.powf(3.0 / 2.0) * g.sqrt() + 2.0 * etav - etav.abs()) / (2.0 * eta),
        ],
    ]
}
