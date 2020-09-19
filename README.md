# What is this?

This a collection of Summation By Parts operators which can be used for solving partial differential equations. The operators has special properties that puts extra emphasis on boundary treatments, ensuring stable (energy bounded) and accurate (4th and 8th order) solutions to some PDEs.

This is a Rust reimplementation [reimplementation](https://gitlab.com/mulimoen/EulerSolver) of code written during my Masters Thesis, made more general and easier to apply to more equations and cases.

## Multigrid
This is the frontend for the Compressible Euler Equations, allowing solving the equations on curvilinear multi-block domains. The solver can be run in parallel using a block-per-thread approach (utilising `rayon`).


## Webfront
This is a test at combining rust+WASM+WebGL+SBP. The prototypes can be seen [here (Maxwell solver)](https://ulimoen.dev/physics/websbp/maxwell), [here (Nonlinear Euler solver)](https://ulimoen.dev/physics/websbp/euler), and [here (shallow water equations)](https://ulimoen.dev/physics/websbp/shallowWater).

## SBP
The main portion of the library, composed of Traits for the operators, and implementation for some common operators in literature.
