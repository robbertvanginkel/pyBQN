# -*- coding: utf-8 -*-
import functools
import inspect
import itertools
import operator

from .vm import Array, Block, Modifier


def bqnfn(fn):
    """wraps a function that takes a named special argument (sxwrfg) and returns a function that takes a list of argumnts in vm order"""
    kwargs = [(i, s) for i, s in enumerate("sxwrfg") if s in inspect.signature(fn).parameters.keys()]
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
        case Modifier():
            return {Block.Type.N1MOD: 4, Block.Type.N2MOD: 5}[x.type]
        case _:
            if callable(x):
                return 3
            else:
                raise ValueError(f"unknown type {type(x)}")


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
        return Array(x[:], x.shape, w)
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
        raise ValueError(w if w is not None else x)  # TODO: VMError?
    else:
        return x


@bqnfn
def pplus(x, w):
    if w is None:
        if type(x) is not float and type(x) is not int:
            raise Exception("+: 𝕩 must be a number")
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
    if w is None:  # TODO: sub strings/number combo behavior
        return -x
    else:
        return w - x


@bqnfn
def pdivide(x, w):
    w = w if w is not None else 1
    if type(x) == int and type(w) == int:
        floordiv = w // x
        truediv = w / x
        return floordiv if floordiv == truediv else truediv
    else:
        return w / x


@bqnfn
def ppower(x, w):
    raise NotImplementedError("power")


@bqnfn
def pfloor(x, w):
    raise NotImplementedError("floor")


@bqnfn
def pequals(x, w):
    if w is None:
        return len(x.shape) if type(x) is Array else 0
    else:
        return 1 if x == w else 0

@bqnfn
def ptable(x, w, f):
    if w is not None:
        return Array(
            [f([f, xi, wi]) for (xi, wi) in itertools.product(x, w)],
            x.shape + w.shape,
        )
    else:
        return Array([f([f, value, None]) for value in x], x.shape)

@bqnfn
def pscan(x, w, f):
    if not x or x.shape == [0]:
        raise ValueError("`: 𝕩 must have rank at least 1")
    if w is not None:
        raise NotImplementedError("scan with 𝕨")
    else:
        stride = functools.reduce(operator.mul, x.shape[1:], 1)
        result = [None] * len(x)
        for i in range(stride):
            result[i] = x[i]
        for i in range(stride, len(x)):
            result[i] = f([f, x[i], result[i - stride]])
        return Array(result, x.shape, x.fill)

@bqnfn
def pfill_by(x, w, f, g):
    return f([g, x, w])  # https://mlochbaum.github.io/BQN/implementation/vm.html#testing

@bqnfn
def pvalences(x, w, f, g):
    if w is not None:
        return g([g, x, w])
    else:
        return f([f, x, w])


def pcatches(args):
    raise NotImplementedError("catches")


provides = [
    ptype,
    pfill,
    plog,
    pgroup_len,
    pgroup_ord,
    passert_fn,
    pplus,
    pminus,
    bqnfn(lambda x, w: w * x),
    pdivide,
    ppower,
    pfloor,
    pequals,
    bqnfn(lambda x, w: w <= x),
    bqnfn(lambda x: Array(x.shape, fill=0)),
    bqnfn(lambda x, w: Array(x[:], w if w is not None else [len(x)], x.fill)),
    bqnfn(lambda x, w: x[w]),
    bqnfn(lambda x: Array(range(x), fill=0)),
    lambda args: Modifier(Block.Type.N1MOD, ptable, *args),
    lambda args: Modifier(Block.Type.N1MOD, pscan, *args),
    lambda args: Modifier(Block.Type.N2MOD, pfill_by, *args),
    lambda args: Modifier(Block.Type.N2MOD, pvalences, *args),
    pcatches,
]
