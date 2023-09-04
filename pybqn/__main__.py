import sys
from pybqn import VM

if __name__ == "__main__":
    # input = sys.stdin.read()
    input = "•listSys"
    vm = VM()
    result = vm.run(input)
    print(vm.format(result))
