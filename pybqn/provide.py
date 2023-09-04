# -*- coding: utf-8 -*-
import functools
import inspect
import itertools
import math
import operator
from typing import Any

from pybqn.program import Array, BQNError, Block, Modifier, Train2D, Train3D, call


def bqnstr(s: str):
    return Array(s, fill=" ")


def bqnfn(fn):
    """wraps a function that takes a named special argument (sxwrfg) and returns a function that takes a list of argumnts in vm order"""
    params = inspect.signature(fn).parameters.keys()
    kwargs = [(i, s) for i, s in enumerate("sxwrfg") if s in params]

    @functools.wraps(fn)
    def do(args):
        return fn(**{s: args[i] for i, s in kwargs})

    return do


@bqnfn
def ptype(x):
    match x:
        case Array():
            return 0
        case int() | float():
            return 1
        case str():
            assert len(x) == 1  # chars only?
            return 2
        case Modifier(type=Block.Type.N1MOD, bound=False):
            return 4
        case Modifier(type=Block.Type.N2MOD, bound=False):
            return 5
        case _:
            if callable(x):
                return 3
            else:
                raise BQNError(f"unknown type {type(x)}")


def to_fill(value):
    if callable(value):
        return None
    elif type(value) is Array:
        return Array([to_fill(x) for x in value], value.shape, value.fill)
    elif type(value) is int or type(value) is float:
        return 0
    else:
        return " "


@bqnfn
def pfill(x, w):
    if w is not None:
        return Array(x[:], x.shape, to_fill(w))
    elif x.fill is None:
        raise BQNError("fill: 𝕩 does not have a fill value")
    else:
        return x.fill


@bqnfn
def plog(x, w):
    match x, w:
        case 0, 0:
            return math.nan
        case 0, _:
            return math.inf if w is not None and w < 1 else -math.inf
        case _:
            return math.log(x, math.e if w is None else w)


@bqnfn
def pgroup_len(x, w):
    l = functools.reduce(max, x, (w if w is not None else 0) - 1)
    r = [0] * (l + 1)
    for e in x:
        if e >= 0:
            r[e] += 1
    return Array(r, fill=0)


@bqnfn
def pgroup_ord(x, w):
    *s, l = itertools.accumulate(w, initial=0)
    r: list[Any] = [None] * l
    for i, e in enumerate(x):
        if e >= 0:
            r[s[e]] = i
            s[e] += 1
    return Array(r, fill=x.fill)


@bqnfn
def passert_fn(x, w):
    if x != 1:
        raise BQNError(w if w is not None else x)
    else:
        return x


@bqnfn
def pplus(x, w):
    if w is None:
        if type(x) is not float and type(x) is not int:
            raise BQNError("+: 𝕩 must be a number")
        return x
    else:
        match w, x:
            case int() | float(), int() | float():
                return w + x
            case int(), str():
                return chr(ord(x) + w)
            case str(), int():
                return chr(ord(w) + x)
            case str(), str():
                raise BQNError("+: Cannot add two characters")
            case _:
                raise BQNError("+: Cannot add non-data values")


@bqnfn
def pminus(x, w):
    match w, x:
        case None, int() | float():
            return -x
        case int() | float(), int() | float():
            return w - x
        case None | int(), str():
            raise BQNError("-: Can only negate numbers")
        case str(), int():
            try:
                return chr(ord(w) - x)
            except ValueError as e:
                raise BQNError(*e.args) from None
        case str(), str():
            return ord(w) - ord(x)
        case _:
            raise BQNError("-: Cannot subtract non-data values")


@bqnfn
def ptimes(x, w):
    match x, w:
        case int() | float(), int() | float():
            return w * x
        case _:
            raise BQNError("×: Arguments must be numbers")


@bqnfn
def pdivide(x, w):
    w = w if w is not None else 1
    try:
        match x, w:
            case int(), int():
                floordiv = w // x
                truediv = w / x
                return floordiv if floordiv == truediv else truediv
            case int() | float(), int() | float():
                return w / x
            case _:
                raise BQNError("÷: Arguments must be numbers")
    except ZeroDivisionError:
        return math.inf if w > 0 else -math.inf if w < 0 else math.nan


@bqnfn
def ppower(x, w):
    try:
        if w is None:
            return math.exp(x)
        else:
            return w**x
    except TypeError as e:
        raise BQNError(*e.args) from None


@bqnfn
def pfloor(x, w):
    try:
        if w is None:
            return math.floor(x) if math.isfinite(x) else x
        else:
            return min(x, w)
    except TypeError as e:
        raise BQNError(*e.args) from None


@bqnfn
def pequals(x, w):
    if w is None:
        return len(x.shape) if type(x) is Array else 0
    else:
        return int(x == w)


@bqnfn
def plessq(x, w):
    match x, w:
        case str(), str():
            return ord(w) <= ord(x)
        case str(), int() | float():
            return 1
        case int() | float(), str():
            return 0
        case _:
            try:
                return int(w <= x)
            except (ValueError, TypeError) as e:
                raise BQNError(*e.args) from None


@bqnfn
def preshape(x, w):
    return Array(x[:], w if w is not None else [len(x)], x.fill)


@bqnfn
def ptable(x, w, f):
    if w is not None:
        return Array(
            [call(f, xi, wi) for (wi, xi) in itertools.product(w, x)],
            w.shape + x.shape,
        )
    else:
        return Array([call(f, value, None) for value in x], x.shape)


@bqnfn
def pscan(x, w, f):
    if x is None or type(x) is not Array or len(x.shape) == 0:
        raise BQNError("`: 𝕩 must have rank at least 1")
    if w is not None:
        rank = len(w.shape) if type(w) is Array else 0
        if rank + 1 != len(x.shape):
            raise BQNError("`: rank of 𝕨 must be cell rank of 𝕩")
        if type(w) is not Array:
            w = [w]  # No need to be Array, only used below
        elif not all(d == x.shape[1 + i] for (i, d) in enumerate(w.shape)):
            raise BQNError("`: shape of 𝕨 must be cell shape of 𝕩")
    if len(x) > 0:
        stride = functools.reduce(operator.mul, x.shape[1:], 1)
        result: list[Any] = [None] * len(x)
        for i in range(stride):
            result[i] = x[i] if w is None else call(f, x[i], w[i])
        for i in range(stride, len(x)):
            result[i] = call(f, x[i], result[i - stride])
        return Array(result, x.shape, x.fill)
    else:
        return Array(x[:], x.shape, x.fill)


@bqnfn
def pfill_by(x, w, f, g):
    r = call(f, x, w)  # https://mlochbaum.github.io/BQN/implementation/vm.html#testing
    # atomfill = lambda x: x if callable(x) else 0 if type(x) in [int, float] else " "
    # xf = x.fill if type(x) is Array else atomfill(x)
    # if type(r) == Array and xf:
    #     r = Array(r[:], r.shape)
    #     try:
    #         wf = None
    #         if w is None:
    #             wf = wf
    #         elif type(w) is not Array:
    #             wf = atomfill(w)
    #         elif type(w) is Array and w.fill is not None:
    #             wf = w.fill
    #         else:
    #             wf = passert_fn
    #         r.fill = to_fill(g([g, xf, wf]))
    #     except AssertionError:
    #         r.fill = None
    return r


@bqnfn
def pvalences(x, w, f, g):
    if w is not None:
        return call(g, x, w)
    else:
        return call(f, x, w)


@bqnfn
def pcatches(x, w, f, g):
    try:
        return call(f, x, w)
    except BQNError:
        return call(g, x, w)


provides = [
    ptype,
    pfill,
    plog,
    pgroup_len,
    pgroup_ord,
    passert_fn,
    pplus,
    pminus,
    ptimes,
    pdivide,
    ppower,
    pfloor,
    pequals,
    plessq,
    bqnfn(lambda x: Array(x.shape, fill=0)),
    preshape,
    bqnfn(lambda x, w: x[w]),
    bqnfn(lambda x: Array(range(x), fill=0)),
    Modifier(Block.Type.N1MOD, ptable),
    Modifier(Block.Type.N1MOD, pscan),
    Modifier(Block.Type.N2MOD, pfill_by),
    Modifier(Block.Type.N2MOD, pvalences),
    pcatches,
]


def make_prims(runtime):
    @bqnfn
    def decompose(x):
        if x in runtime:
            return Array([0, x])
        match x:
            case Array() | int() | float() | str():
                return Array([-1, x])
            case Train2D():
                return Array([2, x.G, x.H])
            case Train3D():
                return Array([3, x.F, x.G, x.H])
            case Modifier(type=Block.Type.N1MOD, bound=True):
                return Array([4, x.f, x.r])
            case Modifier(type=Block.Type.N2MOD, bound=True):
                return Array([5, x.f, x.r, x.g])
            case _:
                return Array([1, x])

    index = {id(x): i for i, x in enumerate(runtime)}

    @bqnfn
    def prim_ind(x):
        return index.get(id(x), len(runtime))

    @bqnfn
    def glyph(x):
        idx = index[id(x)]
        return "+-×÷⋆√⌊⌈|¬∧∨<>≠=≤≥≡≢⊣⊢⥊∾≍⋈↑↓↕«»⌽⍉/⍋⍒⊏⊑⊐⊒∊⍷⊔!˙˜˘¨⌜⁼´˝`∘○⊸⟜⌾⊘◶⎉⚇⍟⎊"[idx]

    return decompose, prim_ind, glyph


@bqnfn
def fmt_num(x):
    return str(x)
