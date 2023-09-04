import sys
from typing import Any, Optional, TextIO
from pybqn.program import BQNError, Program, call, Array
from pybqn.provide import make_prims, provides, ptype, fmt_num, bqnstr
from pybqn.precompiled import r, c, f


def unstr(x: Array) -> str:
    return "".join(x)


class System:
    def __init__(self, **kwargs):
        self.funcs = kwargs
        if len(self.funcs) > 0:
            self.funcs["listsys"] = Array([bqnstr(k) for k in self.funcs.keys()])

    def __call__(self, _s, x, _w):
        try:
            return Array([self.funcs[unstr(func_bqnstr)] for func_bqnstr in x])
        except KeyError as e:
            raise BQNError(
                f"Unknown system values (see •listSys for available): •{e.args[0]}"
            ) from None


class VM:
    def __init__(
        self, args: Optional[list[str]] = None, stdout: Optional[TextIO] = sys.stdout
    ):
        self._runtime, set_prims, _ = Program(*r(provides))()
        decompose, prim_ind, glyph = make_prims(self._runtime)
        call(set_prims, Array([decompose, prim_ind]))
        self._compiler = Program(*c(self._runtime))()
        formatter = Program(*f(self._runtime))()
        self._fmt, self._repr = call(
            formatter, Array([ptype, formatter, glyph, fmt_num])
        )
        syskwargs = {}
        if args is not None:
            syskwargs["args"] = Array([bqnstr(arg) for arg in args])
        if stdout is not None:

            def write(_s, x, _w):
                if type(x) is not Array:
                    raise BQNError("Trying to output non-character")
                stdout.write(unstr(x))
                stdout.write("\n")
                return x

            syskwargs["out"] = write
        self._system = System(
            fmt=self._fmt,
            repr=self._repr,
            **syskwargs,
        )

    def compile(self, source: str) -> Program:
        compiled = call(
            self._compiler, x=bqnstr(source), w=Array([self._runtime, self._system])
        )
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
        reprd = call(self._repr, result)
        if reprd is not None:
            return unstr(reprd)
        return ""
