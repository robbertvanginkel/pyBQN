import argparse
import os
import select
import sys
import pybqn
from pybqn.program import Array
from pybqn.provide import bqnstr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pybqn")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "file",
        nargs="*",
        help="bqn file to execute",
        metavar="file.bqn args",
        default=[None],
    )
    input_group.add_argument("-p", metavar="expression", help="execute as bqn and pretty print result")
    parser.add_argument("--version", action="version", version=pybqn.__version__)
    args = parser.parse_args()
    file, *fileargs = args.file

    path, filename = os.path.join(os.getcwd(), ""), "."
    if file is not None:
        with open(file) as f:
            input = f.read()
        path = os.path.dirname(file)
        filename = os.path.basename(file)
    elif args.p is not None:
        input = args.p
    elif select.select([sys.stdin, ], [], [], 0.0)[0]:  # fmt: skip
        input = sys.stdin.read()
    else:
        input = input("pybqn> ")

    vm = pybqn.VM(stdout=sys.stdout)
    result = vm.run(
        bqnstr(input),
        Array(
            [
                bqnstr(path),
                bqnstr(filename),
                Array([bqnstr(x) for x in fileargs]),
            ]
        ),
    )
    if args.file == [None]:
        print(vm.format(result))
