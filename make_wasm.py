#! /usr/bin/env python3

from argparse import ArgumentParser
from subprocess import call

if __name__ == "__main__":
    parser = ArgumentParser(description="Build js and wasm")
    parser.add_argument("-r", help="Build release type",
                        dest="release", action="store_true")

    args = parser.parse_args()

    if args.release:
        call(["cargo", "build", "--release",
              "--target", "wasm32-unknown-unknown"])
        target = "target/wasm32-unknown-unknown/release/webgl.wasm"
    else:
        call(["cargo", "build",
              "--target", "wasm32-unknown-unknown"])
        target = "target/wasm32-unknown-unknown/debug/webgl.wasm"

    call(["wasm-bindgen", target, "--out-dir", ".",
          "--no-typescript", "--target", "web"])
