from typing import Any
from pybqn.program import BQNError, Program, call, Array
from pybqn.provide import make_prims, provides, ptype, fmt_num, bqnstr
from pybqn.precompiled import r, c, f


class VM:
    def __init__(self):
        self._runtime, set_prims, _ = Program(*r(provides))()
        decompose, prim_ind, glyph = make_prims(self._runtime)
        call(set_prims, Array([decompose, prim_ind]))
        self._compiler = Program(*c(self._runtime))()
        formatter = Program(*f(self._runtime))()
        self._fmt, _ = call(formatter, Array([ptype, formatter, glyph, fmt_num]))

    def compile(self, source: str) -> Program:
        compiled = call(self._compiler, x=bqnstr(source), w=self._runtime)
        return Program(*compiled)

    def run(self, source: str) -> Any:
        try:
            return self.compile(source)()
        except BQNError as e:
            if type(e.args[0]) is Array:
                raise BQNError(self.format(e.args[0])) from None
            else:
                raise

    def format(self, result: Any) -> str:
        return "".join(call(self._fmt, result))
