# -*- coding: utf-8 -*-
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable, List, Optional
import functools
import itertools


# https://gist.github.com/dzaima/e7b24e10cf6ac33f62bf8cfd80758d4b


class Array(list):
    shape: list[int]
    fill: Any

    def __init__(
        self, list: Iterable, shape: Optional[list[int]] = None, fill: Any = None
    ):
        super().__init__(list)
        self.shape = shape if shape is not None else [len(self)]
        self.fill = (
            fill
            if fill is not None
            else 0
            if len(list) > 0 and all(type(x) in [int, float] for x in self)
            else None
        )


def unstr(x: Array) -> str:
    return "".join(x)


class Stack(list):
    def popn(self, n: int):
        if n == 0:
            return []
        if n > len(self):
            raise IndexError(f"popn n={n} from with len()={len(self)}")
        as_list = self[-n:]
        del self[-n:]
        return as_list


class partial(functools.partial):
    """https://github.com/python/cpython/issues/65329"""

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.func == other.func
            and self.args == other.args
            and self.keywords == other.keywords
        )


@dataclass
class Train2D:
    G: Any
    H: Any

    def __call__(self, *args):
        return call(self.G, call(self.H, args[1], args[2]))


@dataclass
class Train3D:
    G: Any
    H: Any
    F: Any

    def __call__(self, *args):
        return call(
            self.G,
            call(self.H, args[1], args[2]),
            call(self.F, args[1], args[2]) if self.F is not None else None,
        )


@dataclass
class Slot:
    cleared: bool = field(init=False, default=False)
    _set: bool = field(init=False, default=False)
    _value: Any = field(init=False, default=object())  # must be not None

    @property
    def value(self):
        if not self.set or self.cleared:
            raise Exception(f"invalid slot access {self}")
        return self._value

    @value.setter
    def value(self, value):
        if type(value) is Slot:
            value = value.value
        self._set = True
        self._value = value

    @value.deleter
    def value(self):
        self.cleared = True
        del self._value

    @staticmethod
    def set(
        ref: Optional["Slot"] | List[Optional["Slot"]],
        value: Any | list,
        allow_uninitialized=False,
    ):
        if type(ref) is Slot:
            if not allow_uninitialized and ref._set is False:
                raise Exception("uninitialized variable")
            ref.value = value
        elif type(ref) is Array:
            if type(value) is not Array or len(ref) != len(value):
                raise BQNError("assignment lenght mismatch")
            for s, v in zip(ref, value):
                Slot.set(s, v, allow_uninitialized)
        elif ref is None:
            pass  # TODO: double check this
        else:
            raise Exception(f"Unknown slot type {ref}")

    @staticmethod
    def get(ref: Optional["Slot"] | List[Optional["Slot"]]):
        if type(ref) is Slot:
            return ref.value
        elif ref is None:
            return None
        elif type(ref) is Array:
            return Array([Slot.get(x) for x in ref])


class Frame:
    depth: int = 0

    def __init__(self, parent: Optional["Frame"], nvars: int, *args) -> None:
        self.depth = parent.depth + 1 if parent else 0
        self.parent = parent
        self.slots = [Slot() for _ in range(nvars)]
        for i, arg in enumerate(args[:nvars]):
            self.slots[i].value = arg

    def slot(self, n_up: int, index: int):
        if n_up == 0:
            return self.slots[index]
        elif self.parent is not None:
            return self.parent.slot(n_up - 1, index)
        else:
            raise IndexError()


@dataclass
class Block:
    class Immediateness(IntEnum):
        DEFERRED = 0
        IMMEDIATE = 1

    class Type(IntEnum):
        FUNIMM = 0
        N1MOD = 1
        N2MOD = 2

    prog: "Program"
    block_type: Type
    block_immediate: Immediateness
    index: int | list[list[int]]

    def __call__(self, parent: Optional[Frame] = None):
        runner: Callable[..., Any]
        match self.block_type:
            case Block.Type.FUNIMM:
                runner = partial(self.run_bc, parent)
            case Block.Type.N1MOD | Block.Type.N2MOD:
                runner = Modifier(
                    self.block_type,
                    partial(self.run_bc, parent),
                )
            case _:
                raise Exception(f"unknonwn block_type {self}")

        match self.block_immediate:
            case Block.Immediateness.IMMEDIATE:
                return runner()
            case Block.Immediateness.DEFERRED:
                return runner
            case _:
                raise Exception(f"unknonwn block_immediate {self}")

    def run_bc(self, parent: Optional[Frame], *args):
        if type(self.index) is int:
            return self.prog.bodies[self.index](parent, *args)
        elif type(self.index) is list or type(self.index) is Array:
            _, _, w, *_ = args
            i = 0 if w is None else 1
            if len(self.index[i]) > 1:
                raise NotImplementedError("multiple body blocks")  # TODO: all blocks
            return self.prog.bodies[self.index[i][0]](parent, *args)
        else:
            raise Exception(f"unreachable {type(self.index)}")


@dataclass
class Body:
    prog: "Program"
    start: int
    nvars: int
    names: list[int] = field(default_factory=list)
    exported: list[bool] = field(default_factory=list)

    def __call__(self, parent: Optional[Frame], *args):
        frame = Frame(parent, self.nvars, *args)
        pc = self.start
        stack = Stack()
        while True:
            opcode = self.prog.bc[pc]
            # logging.debug(f"pc: {pc:02d}, op: 0x{opcode:02x}/{opcode}, stack: {stack}")
            match opcode:
                case 0x00:  # PUSH
                    arg = self.prog.bc[pc + 1]
                    stack.append(self.prog.constants[arg])
                    pc += 2
                case 0x01:  # DFND
                    arg = self.prog.bc[pc + 1]
                    stack.append(self.prog.blocks[arg](frame))
                    pc += 2
                case 0x06:  # POPS
                    _ = stack.pop()
                    pc += 1
                case 0x07:  # RETN
                    return stack.pop()
                case 0x08:  # RETD
                    ns = {
                        k: v
                        for k, v in zip(
                            (unstr(self.prog.names[i]) for i in self.names),
                            itertools.compress(frame.slots[-len(self.exported):], self.exported),
                        )
                    }
                    return ns
                case 0x0B | 0x0C:  # LSTO, LSTM
                    stack.append(Array(stack.popn(self.prog.bc[pc + 1])))
                    pc += 2
                case 0x10:  # FN1C
                    x, S = stack.popn(2)
                    if x is None:
                        raise Exception("𝕩 may not be ·")
                    stack.append(call(S, x))
                    pc += 1
                case 0x11:  # FN2C
                    x, S, w = stack.popn(3)
                    if w is None:
                        raise Exception("𝕨 may not be ·")
                    if x is None:
                        raise Exception("𝕨 may not be ·")
                    stack.append(call(S, x, w))
                    pc += 1
                case 0x12:  # FN1O
                    x, S = stack.popn(2)
                    stack.append(call(S, x))
                    pc += 1
                case 0x13:  # FN2O
                    x, S, w = stack.popn(3)
                    stack.append(call(S, x, w))
                    pc += 1
                case 0x14:  # TR2D
                    H, G = stack.popn(2)
                    stack.append(Train2D(G, H))
                    pc += 1
                case 0x15 | 0x17:  # TR3D, TR3O
                    H, G, F = stack.popn(3)
                    if F is None and opcode == 0x15:
                        raise Exception("𝔽 may not be ·")
                    stack.append(Train3D(G, H, F))
                    pc += 1
                case 0x16:  # CHKV
                    raise NotImplementedError
                case 0x1A:  # MD1C
                    R, F = stack.popn(2)
                    stack.append(call(R, F))
                    pc += 1
                case 0x1B:  # MD2C
                    G, R, F = stack.popn(3)
                    stack.append(call(R, F, G))
                    pc += 1
                case 0x20:  # VARO
                    D, I = self.prog.bc[pc + 1 : pc + 3]
                    stack.append(frame.slot(D, I).value)
                    pc += 3
                case 0x21:  # VARM
                    D, I = self.prog.bc[pc + 1 : pc + 3]
                    stack.append(frame.slot(D, I))
                    pc += 3
                case 0x22:  # VARU
                    D, I = self.prog.bc[pc + 1 : pc + 3]
                    slot = frame.slot(D, I)
                    stack.append(slot.value)
                    del slot.value
                    pc += 3
                case 0x2C:  # NOTM
                    stack.append(None)  # TODO: Dubious
                    pc += 1
                case 0x30:  # SETN
                    val, ref = stack.popn(2)
                    Slot.set(ref, val, allow_uninitialized=True)
                    stack.append(val)
                    pc += 1
                case 0x31:  # SETU
                    val, ref = stack.popn(2)
                    Slot.set(ref, val, allow_uninitialized=False)
                    stack.append(val)
                    pc += 1
                case 0x32:  # SETM
                    x, F, r = stack.popn(3)
                    Slot.set(r, call(F, x, Slot.get(r)))
                    stack.append(Slot.get(r))
                    pc += 1
                case 0x33:  # SETC
                    F, r = stack.popn(2)
                    Slot.set(r, call(F, Slot.get(r)))
                    stack.append(Slot.get(r))
                    pc += 1
                case 0x40:  # FLDO
                    i = self.prog.bc[pc + 1]
                    ns = stack.pop()
                    name = unstr(self.prog.names[i])
                    try:
                        stack.append(ns[name])
                    except TypeError:
                        raise BQNError(
                            f"key lookup is not a namespace '{type(ns)}'"
                        ) from None
                    except KeyError:
                        raise BQNError(f"unknown namespace key'{name}'") from None
                    pc += 2
                case _:
                    raise Exception(
                        f"Unknown opcode at {pc}: 0x{opcode:02x}, top10stack: {stack[-10:]}"
                    )


def call(F, x=None, w=None) -> Any:
    if x is None:
        return None
    if callable(F):
        return F(F, x, w)
    else:
        return F


@dataclass
class Modifier:
    type: Block.Type
    callable: Callable
    bound: bool = False
    r: Any = None
    f: Any = None
    g: Any = None

    def __call__(self, *args):
        if not self.bound:
            return Modifier(self.type, self.callable, True, *args)
        else:
            return self.callable(*args, self.r, self.f, self.g)


class BQNError(Exception):
    """raised when execution of a BQN program throws an error"""


class Program:
    def __init__(
        self, bc: list[int], constants: list[Any], blocks, bodies, source=None, tok=None
    ) -> None:
        self.bc = bc
        self.constants = constants
        self.bodies = [Body(self, *x) for x in bodies]
        self.blocks = [Block(self, *x) for x in blocks]
        self.source = source
        self.tok = tok
        if self.tok is not None:
            self.names = self.tok[2][0]

    def __call__(self) -> Any:
        return self.blocks[0]()
