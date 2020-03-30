#! /usr/bin/env python3

from argparse import ArgumentParser
from subprocess import check_call
from shutil import copyfile, copytree
import tempfile
import pathlib
import os

if __name__ == "__main__":
    parser = ArgumentParser(description="Build js and wasm")
    parser.add_argument(
        "-r", help="Build release type", dest="release", action="store_true"
    )
    parser.add_argument(
        "--destdir",
        default=pathlib.Path("publish"),
        type=pathlib.Path,
        help="Destination suitable for being copied directly to the webserver",
    )

    args = parser.parse_args()

    publish = args.destdir
    publish.mkdir(exist_ok=True)

    target_triple = "wasm32-unknown-unknown"
    command = ["cargo", "build", "--target", target_triple]
    env = os.environ.copy()
    if args.release:
        env["RUSTFLAGS"] = "-C opt-level=3 -C codegen-units=1 -C lto=fat"
        command.append("--release")

    check_call(command, env=env)

    target = (
        pathlib.Path("../target")
        .joinpath(target_triple)
        .joinpath("release" if args.release else "debug")
        .joinpath("sbp_web.wasm")
    )
    assert target.exists()

    check_call(
        [
            "wasm-bindgen",
            str(target),
            "--out-dir",
            str(publish),
            "--no-typescript",
            "--target",
            "web",
        ]
    )

    if args.release:
        try:
            with tempfile.TemporaryDirectory() as d_:
                d = pathlib.Path(d_)
                wasm_bg = publish.joinpath("sbp_web_bg.wasm")
                wasm_to_opt = d.joinpath("before-wasm-opt.wasm")
                copyfile(wasm_bg, wasm_to_opt)
                check_call(["wasm-opt", "-O4", str(wasm_to_opt), "-o", str(wasm_bg)])
        except FileNotFoundError:
            print("wasm-opt not found, not optimising further")
            pass

    copytree("pages", publish, dirs_exist_ok=True)
