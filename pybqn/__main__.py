import argparse
import select
import sys
import pybqn

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
    vm = pybqn.VM(args=args, stdout=sys.stdout)

    if file is not None:
        with open(file) as f:
            input = f.read()
    elif select.select([sys.stdin, ], [], [], 0.0)[0]:  # fmt: skip
        input = sys.stdin.read()
    else:
        input = input("pybqn> ")
    result = vm.run(input)
    print(vm.format(result))
