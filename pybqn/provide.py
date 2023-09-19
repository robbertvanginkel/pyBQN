# -*- coding: utf-8 -*-
import functools
import itertools
import math
import operator
from typing import Any

from pybqn.program import Array, BQNError, Block, Modifier, Train2D, Train3D, call


def bqnstr(s: str):
    return Array(s, fill=" ")


def ptype(_s, x, _w):
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
        case dict():
            return 6
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


def pfill(_s, x, w):
    if w is not None:
        return Array(x[:], x.shape, to_fill(w))
    elif x.fill is None:
        raise BQNError("fill: 𝕩 does not have a fill value")
    else:
        return x.fill


def plog(_s, x, w):
    match x, w:
        case 0, 0:
            return math.nan
        case 0, _:
            return math.inf if w is not None and w < 1 else -math.inf
        case _:
            return math.log(x, math.e if w is None else w)


def pgroup_len(_s, x, w):
    l = functools.reduce(max, x, (w if w is not None else 0) - 1)
    r = [0] * (l + 1)
    for e in x:
        if e >= 0:
            r[e] += 1
    return Array(r, fill=0)


def pgroup_ord(_s, x, w):
    *s, l = itertools.accumulate(w, initial=0)
    r: list[Any] = [None] * l
    for i, e in enumerate(x):
        if e >= 0:
            r[s[e]] = i
            s[e] += 1
    return Array(r, fill=x.fill)


def passert_fn(_s, x, w):
    if x != 1:
        raise BQNError(w if w is not None else x)
    else:
        return x


def pplus(_s, x, w):
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


def pminus(s_, x, w):
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


def ptimes(s_, x, w):
    if x == 0 or w == 0: 
        return 0 # see testvm.TestVM.test_pi
    match x, w:
        case int() | float(), int() | float():
            return w * x
        case _:
            raise BQNError("×: Arguments must be numbers")


def pdivide(_s, x, w):
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


def ppower(_s, x, w):
    try:
        if w is None:
            return math.exp(x)
        else:
            return w**x
    except TypeError as e:
        raise BQNError(*e.args) from None


def pfloor(_s, x, w):
    try:
        if w is None:
            return math.floor(x) if math.isfinite(x) else x
        else:
            return min(x, w)
    except TypeError as e:
        raise BQNError(*e.args) from None


def pequals(_s, x, w):
    if w is None:
        return len(x.shape) if type(x) is Array else 0
    else:
        return int(x == w)


def plessq(_s, x, w):
    match x, w:
        case str(), str():
            return int(ord(w) <= ord(x))
        case str(), int() | float():
            return 1
        case int() | float(), str():
            return 0
        case _:
            try:
                return int(w <= x)
            except (ValueError, TypeError) as e:
                raise BQNError(*e.args) from None


def preshape(_s, x, w):
    return Array(x[:], w if w is not None else [len(x)], x.fill)


def ptable(_s, x, w, _r, f, _g):
    if w is not None:
        return Array(
            [call(f, xi, wi) for (wi, xi) in itertools.product(w, x)],
            w.shape + x.shape,
        )
    else:
        return Array([call(f, value, None) for value in x], x.shape)


def pscan(_s, x, w, _r, f, _g):
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


def pfill_by(_s, x, w, _r, f, g):
    r = call(f, x, w)  # https://mlochbaum.github.io/BQN/implementation/vm.html#testing
    atomfill = lambda x: x if callable(x) else 0 if type(x) in [int, float] else " "
    xf = x.fill if type(x) is Array else atomfill(x)
    if type(r) == Array and xf is not None:
        r = Array(r[:], r.shape)
        try:
            wf = None
            if w is None:
                wf = wf
            elif type(w) is not Array:
                wf = atomfill(w)
            elif type(w) is Array and w.fill is not None:
                wf = w.fill
            else:
                wf = passert_fn
            r.fill = to_fill(call(g, xf, wf))
        except BQNError:
            r.fill = None
    return r


def pvalences(_s, x, w, _r, f, g):
    if w is not None:
        return call(g, x, w)
    else:
        return call(f, x, w)


def pcatches(_s, x, w, _r, f, g):
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
    lambda _s, x, _w: Array(x.shape, fill=0),
    preshape,
    lambda _s, x, w: x[w],
    lambda _s, x, _w: Array(range(x), fill=0),
    Modifier(Block.Type.N1MOD, ptable),
    Modifier(Block.Type.N1MOD, pscan),
    Modifier(Block.Type.N2MOD, pfill_by),
    Modifier(Block.Type.N2MOD, pvalences),
    Modifier(Block.Type.N2MOD, pcatches),
]


def make_prims(runtime):
    def decompose(_s, x, _w):
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

    def prim_ind(_s, x, _w):
        return index.get(id(x), len(runtime))

    def glyph(_s, x, _w):
        idx = index[id(x)]
        return "+-×÷⋆√⌊⌈|¬∧∨<>≠=≤≥≡≢⊣⊢⥊∾≍⋈↑↓↕«»⌽⍉/⍋⍒⊏⊑⊐⊒∊⍷⊔!˙˜˘¨⌜⁼´˝`∘○⊸⟜⌾⊘◶⎉⚇⍟⎊"[idx]

    return decompose, prim_ind, glyph


def fmt_num(_s, x, _w):
    return bqnstr(str(x))
