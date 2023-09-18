import argparse
import os
import select
import sys
import pybqn
from pybqn.program import Array
from pybqn.provide import bqnstr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pybqn")
    parser.add_argument(
        "file",
        nargs="*",
        help="bqn file to execute",
        metavar="file.bqn args",
        default=[None],
    )
    parser.add_argument("--version", action="version", version=pybqn.__version__)
    args = parser.parse_args()
    file, *args = args.file
    vm = pybqn.VM(stdout=sys.stdout)

    path, filename = os.path.join(os.getcwd(), ""), "."
    if file is not None:
        with open(file) as f:
            input = f.read()
        path = os.path.dirname(file)
        filename = os.path.basename(file)
    elif select.select([sys.stdin, ], [], [], 0.0)[0]:  # fmt: skip
        input = sys.stdin.read()
    else:
        input = input("pybqn> ")
    result = vm.run(
        bqnstr(input),
        Array(
            [
                bqnstr(path),
                bqnstr(filename),
                Array([bqnstr(x) for x in args]),
            ]
        ),
    )
    print(result)
