from typing import Any
from pybqn.program import Program, call, Array
from pybqn.provide import make_prims, provides, ptype, fmt_num, bqnstr
from pybqn.precompiled import r, c, f


class VM:
    def __init__(self):
        runtime, set_prims, _ = Program(*r(provides))()
        decompose, prim_ind, glyph = make_prims(runtime)
        call(set_prims, Array([decompose, prim_ind]))
        compiler = Program(*c(runtime))()
        formatter = Program(*f(runtime))()
        self._fmt, _ = call(formatter, Array([ptype, formatter, glyph, fmt_num]))
        self._runtime = runtime
        self._compiler = compiler

    def compile(self, source: str) -> Program:
        compiled = call(self._compiler, x=bqnstr(source), w=self._runtime)
        return Program(*compiled)

    def run(self, source: str) -> Any:
        return self.compile(source)()

    def format(self, result: Any) -> str:
        return "".join(call(self._fmt, result))
