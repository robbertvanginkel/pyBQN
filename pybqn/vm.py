import functools
import os
from dataclasses import dataclass
from posixpath import isabs
from typing import Any, Optional, TextIO

from pybqn.precompiled import c, f, r
from pybqn.program import Array, BQNError, Program, call, unstr
from pybqn.provide import bqnstr, fmt_num, make_prims, provides, ptype


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


@dataclass
class File:
    path: str

    def full_path(self, x: str) -> str:
        if not isabs(x):
            return os.path.join(self.path, x)
        return x

    def file_list(self, _s, x, _w):
        return Array([bqnstr(f) for f in os.listdir(self.full_path(unstr(x)))])

    def file_lines(self, _s, x, w):
        if w is not None:
            raise NotImplementedError("lines: 𝕨 not implemented")
        with open(self.full_path(unstr(x))) as f:
            return Array([bqnstr(line) for line in f.read().splitlines()])

    def __getitem__(self, item):
        return {"lines": self.file_lines, "list": self.file_list}[item]


def sys_write(stdout: TextIO, _s, x, _w):
    if type(x) is not Array:
        raise BQNError("Trying to output non-character")
    stdout.write(unstr(x))
    stdout.write("\n")
    return x


class VM:
    def __init__(self, stdout: Optional[TextIO] = None):
        self._runtime, set_prims, _ = Program(*r(provides))()
        decompose, prim_ind, glyph = make_prims(self._runtime)
        call(set_prims, Array([decompose, prim_ind]))
        self._compiler = Program(*c(self._runtime))()
        formatter = Program(*f(self._runtime))()
        self._fmt, self._repr = call(
            formatter, Array([ptype, formatter, glyph, fmt_num])
        )
        self.stdout = stdout

    def compile(self, source: Array, state: Array) -> Program:
        path = state[0] if state and len(state) > 0 else Array([])
        name = state[1] if state and len(state) > 1 else Array([])
        args = state[2] if state and len(state) > 2 else Array([])
        syskwargs = {}
        if self.stdout is not None:
            syskwargs["out"] = functools.partial(sys_write, self.stdout)

        system = System(
            fmt=self._fmt,
            repr=self._repr,
            file=File(path=unstr(path)),
            bqn=lambda _s, x, w: self.run(x, w),
            state=Array([path, name, args]),
            path=path,
            name=name,
            args=args,
            wdpath=bqnstr(os.path.join(os.getcwd(), "")),
            **syskwargs,
        )
        compiled = call(self._compiler, x=source, w=Array([self._runtime, system]))
        return Program(*compiled)

    def run(self, source: Array, state: Array) -> Any:
        try:
            return self.compile(source, state)()
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
