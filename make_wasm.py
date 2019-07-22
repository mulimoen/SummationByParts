#! /usr/bin/env python3

from argparse import ArgumentParser
from subprocess import check_call
from shutil import copyfile

if __name__ == "__main__":
    parser = ArgumentParser(description="Build js and wasm")
    parser.add_argument("-r", help="Build release type",
                        dest="release", action="store_true")

    args = parser.parse_args()

    if args.release:
        check_call(["cargo", "build", "--release",
                    "--target", "wasm32-unknown-unknown"])
        target = "target/wasm32-unknown-unknown/release/webgl.wasm"
    else:
        check_call(["cargo", "build",
                    "--target", "wasm32-unknown-unknown"])
        target = "target/wasm32-unknown-unknown/debug/webgl.wasm"

    check_call(["wasm-bindgen", target, "--out-dir", ".",
                "--no-typescript", "--target", "web"])

    if args.release:
        try:
            copyfile("webgl_bg.wasm", "before-wasm-opt.wasm")
            check_call(["wasm-opt", "-O4", "before-wasm-opt.wasm",
                        "-o", "webgl_bg.wasm"])
        except FileNotFoundError:
            print("wasm-opt not found, not optimising further")
            pass
