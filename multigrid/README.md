# Multigrid

The input is `json5` files, superset of `json`, but supports comments and trailing commas. Examples of configuration files can be found under `examples`, and most combinations of options are found running the `output_configuration` test:
```sh
cargo test output_configuration -- --nocapture
```

Output is on the `hdf5` format, using a group-based system for storing the solution. Use `eulerplot` to plot the solution.
