# What is it?

This is a test at combining rust+WASM+WebGL+SBP. The prototypes can be seen [here (Maxwell solver)](https://ulimoen.dev/physics/websbp/maxwell), [here (Nonlinear Euler solver)](https://ulimoen.dev/physics/websbp/euler), and [here (shallow water equations)](https://ulimoen.dev/physics/websbp/shallowWater).

# Building
Run `make_wasm.py` or `make_wasm.py -r` for the release version.

# Running
After building, a http server needs to serve the contents. WASM requires this server to serve files ending with `.wasm` as the `application/wasm` MIME type.
