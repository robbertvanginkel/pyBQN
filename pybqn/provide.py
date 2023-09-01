# -*- coding: utf-8 -*-
import functools
import inspect
import itertools
import math
import operator

from .vm import Array, Block, Modifier, Train2D, Train3D, call


def bqnfn(fn):
    """wraps a function that takes a named special argument (sxwrfg) and returns a function that takes a list of argumnts in vm order"""
    params = inspect.signature(fn).parameters.keys()
    kwargs = [(i, s) for i, s in enumerate("sxwrfg") if s in params]

    @functools.wraps(fn)
    def call(args):
        return fn(**{s: args[i] for i, s in kwargs})

    return call


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
        case Modifier(type=Block.Type.N1MOD):
            return 4
        case Modifier(type=Block.Type.N2MOD):
            return 5
        case _:
            if callable(x):
                return 3
            else:
                raise TypeError(f"unknown type {type(x)}")


@bqnfn
def pfill(x, w):
    def to_fill(value):
        if callable(value):
            return None
        elif type(value) is Array:
            return Array([to_fill(x) for x in value], value.shape, value.fill)
        elif type(value) is int or type(value) is float:
            return 0
        else:
            return " "

    if w is not None:
        return Array(x[:], x.shape, to_fill(w))
    elif x.fill is None:
        raise ValueError("fill: 𝕩 does not have a fill value")
    else:
        return x.fill


@bqnfn
def plog(x, w):
    raise NotImplementedError("log")


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
    raise NotImplementedError("log")


@bqnfn
def passert_fn(x, w):
    if x != 1:
        raise AssertionError("".join(w) if w is not None else x)  # TODO: VMError, displaying bqnstr as string
    else:
        return x


@bqnfn
def pplus(x, w):
    if w is None:
        if type(x) is not float and type(x) is not int:
            raise TypeError("+: 𝕩 must be a number")
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
                raise TypeError("+: Cannot add two characters")
            case _:
                raise TypeError("+: Cannot add non-data values")


@bqnfn
def pminus(x, w):
    match w, x:
        case None, int() | float():
            return -x
        case int() | float(), int() | float():
            return w - x
        case None | int(), str():
            raise TypeError("-: Can only negate numbers")
        case str(), int():
            return chr(ord(w) - x)
        case str(), str():
            return ord(w) - ord(x)
        case _:
            raise TypeError("-: Cannot subtract non-data values")


@bqnfn
def ptimes(x, w):
    match x, w:
        case int() | float(), int() | float():
            return w * x
        case _:
            raise TypeError("×: Arguments must be numbers")


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
                raise TypeError("÷: Arguments must be numbers")
    except ZeroDivisionError:
        return math.inf if w > 0 else -math.inf if w < 0 else math.nan


@bqnfn
def ppower(x, w):
    if w is None:
        return math.exp(x)
    else:
        return w**x


@bqnfn
def pfloor(x, w):
    if w is None:
        return math.floor(x) if math.isfinite(x) else x
    else:
        return min(x, w)


@bqnfn
def pequals(x, w):
    if w is None:
        return len(x.shape) if type(x) is Array else 0
    else:
        return 1 if x == w else 0


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
            return w <= x


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
    if x is None or len(x.shape) == 0:
        raise ValueError("`: 𝕩 must have rank at least 1")
    if w is not None:
        rank  = len(w.shape) if type(w) is Array else 0
        if rank+1 != len(x.shape):
            raise ValueError("`: rank of 𝕨 must be cell rank of 𝕩")
        if type(w) is not Array:
            w = [w]  # No need to be Array, only used below
        elif not all(d==x.shape[1+i] for (i, d) in enumerate(w.shape)):
            raise ValueError("`: shape of 𝕨 must be cell shape of 𝕩")
    if len(x) > 0:
        stride = functools.reduce(operator.mul, x.shape[1:], 1)
        result = [None] * len(x)
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
    # xf = x.fill if type(x) is Array else x if callable(x) else 0 if type(x) is int or type(x) is float else " "
    # if type(r) == Array and xf:
    #     wf = 0  # TODO
    #     fill = g([g, xf, wf])
    #     r = Array(r[:], r.shape, fill)
    return r


@bqnfn
def pvalences(x, w, f, g):
    if w is not None:
        return call(g, x, w)
    else:
        return call(f, x, w)


def pcatches(args):
    raise NotImplementedError("catches")


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
            # case Modifier(type=Block.Type.N1MOD):
            #     return Array([4, x.f, x.r])
            # case Modifier(type=Block.Type.N2MOD):
            #     return Array([5, x.f, x.r, x.g])
            case _:
                return Array([1, x])

    index = {id(x): i for i, x in enumerate(runtime)}

    @bqnfn
    def prim_ind(x):
        return index.get(id(x), len(runtime))

    return decompose, prim_ind


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
    lambda args: Modifier(Block.Type.N1MOD, ptable, *args),
    lambda args: Modifier(Block.Type.N1MOD, pscan, *args),
    lambda args: Modifier(Block.Type.N2MOD, pfill_by, *args),
    lambda args: Modifier(Block.Type.N2MOD, pvalences, *args),
    pcatches,
]
