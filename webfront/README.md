# Webfront
Running different SBP solvers in the browser using `WASM`.

## Building
Run `make_wasm.py -r` to compile the library, and put the finished artifacts and related pieces in the `publish` directory.

## Running
After building, a http server needs to serve the contents. WASM requires this server to serve files ending with `.wasm` as the `application/wasm` MIME type. Python can be utilised with:
```sh
python3 -m http.server --directory publish
```
A web browser can then be opened on the link suggested.
