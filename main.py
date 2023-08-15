import ast
import select
import sys

from pybqn import VM

if __name__ == "__main__":
    input = ast.literal_eval(sys.stdin.read())
    print(f"""bc:        {input[0]}
constants: {input[1]},
blocks:    {input[2]}, 
bodies:    {input[3]}""")
    print(VM(*input)())
