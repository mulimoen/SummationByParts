# Utilities

## Integrate
The `Integrable` trait, Butcher Tableaus, and functions to integrate explicit partial differential equations. Embedded integration is additionally available, which allows for adaptive time-steps,although selection of timesteps is left to the users.

## Constmatrix
A constant-size matrix which supports matrix-multiplication.


## Float
The basic building-block for selection of the float type. `f32` can be selected for applications requiring lower precision, such as direct visualisation in graphics.

## Fast-Float
A wrapper-type around the unsafe float intrinsics
