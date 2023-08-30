"""
Small scritp to take cases from BQN/test/cases and pre-compile them for use in
unit tests while not yet self hosted.  

pbpaste | python3 utils/testcase.py | column -t -s % | pbcopy
"""
import sys, ast, subprocess

if __name__ == "__main__":
    for x in sys.stdin.readlines():
        line = x.strip()
        expect = 1
        if line.startswith("! % "):
            line = line.removeprefix("! % ")
            expect = "AssertionError"
        compiled = subprocess.run(["bqn", "utils/cpy.bqn", "../../BQN", line], capture_output=True, text=True).stdout.strip()
        print(f"{ast.unparse(ast.Constant(line))} % :({expect}, {compiled}),")
