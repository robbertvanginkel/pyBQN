# -*- coding: utf-8 -*-
from dataclasses import dataclass, field, make_dataclass
from enum import IntEnum
import itertools
from typing import Any, Iterable, List, Optional
from collections.abc import Callable
import functools
import math

# https://gist.github.com/dzaima/e7b24e10cf6ac33f62bf8cfd80758d4b

# constants
_NOTHING = make_dataclass("Nothing", [], frozen=True)() # could be None at this point?

IndexedSlot = make_dataclass("IndexedSlot", [("depth", int), ("idx", int)])

@dataclass
class Slot(IndexedSlot):
    cleared: bool = field(init=False, default=False)
    _set: bool = field(init=False, default=False)
    _value: Any = field(init=False, default=object()) #must be not _NOTHING

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
    def set(ref: "Slot" | List[Optional["Slot"]], value: Any | list, allow_uninitialized=False):
        if type(ref) is Slot:
            if not allow_uninitialized and ref._set is False:
                raise Exception("uninitialized variable")
            ref.value = value
        elif type(ref) is list:
            assert len(ref) == len(value)
            for s, v in zip(ref, value):
                Slot.set(s, v, allow_uninitialized)
        elif ref is _NOTHING:
            pass # TODO: double check this
        else:
            raise Exception(f"Unknown slot type {ref}")

    @staticmethod
    def get(ref: "Slot" | List[Optional["Slot"]]):
        if type(ref) is Slot:
            return ref.value
        elif ref is _NOTHING:
            return _NOTHING
        elif type(ref) is list:
            return [Slot.get(x) for x in ref]

class Frame:
    depth: int = 0
    def __init__(self, parent: Optional['Frame'], nvars: int, args: list) -> None:
        self.depth = parent.depth + 1 if parent else 0
        self.parent = parent
        self.slots = [Slot(self.depth, i) for i in range(nvars)]
        for i, arg in enumerate(args[:nvars]):
            self.slots[i].value = arg

    def slot(self, n_up: int, index: int):
        if n_up == 0:
            return self.slots[index]
        elif self.parent is not None:
            return self.parent.slot(n_up-1, index)
        else:
            raise IndexError()

@dataclass
class Modifier:
    callable: Callable[[list], Any]
    r: Any = _NOTHING
    f: Any = _NOTHING
    g: Any = _NOTHING
    def __call__(self, args: list):
        return self.callable(args + [self.r, self.f, self.g])

class MD1(Modifier): pass
class MD2(Modifier): pass

@dataclass
class Train2D:
    G: Any; H: Any
    def __call__(self, args: list):
        return Body._call(self.G, Body._call(self.H, args[1], args[2]))

@dataclass
class Train3D:
    G: Any; H: Any; F: Any
    def __call__(self, args: list):
        return Body._call(
                self.G,
                Body._call(self.H, args[1], args[2]),
                Body._call(self.F, args[1], args[2]) if self.F is not _NOTHING else _NOTHING,
            )

@dataclass
class Block:
    class Immediateness(IntEnum):
        DEFERRED = 0
        IMMEDIATE = 1
    class Type(IntEnum):
        FUNIMM = 0
        N1MOD = 1
        N2MOD = 2
    vm: 'VM'
    block_type: Type
    block_immediate: Immediateness
    index: int | list[list[int]]

    def __call__(self, parent: Optional[Frame] = None):
        runner: Callable[[Optional[Frame], list], Any]
        match self.block_type:
            case Block.Type.FUNIMM:
                runner = self.run_bc
            case Block.Type.N1MOD:
                runner = lambda parent, modifier_args: MD1(functools.partial(self.run_bc, parent), *modifier_args)
            case Block.Type.N2MOD:
                runner = lambda parent, modifier_args: MD2(functools.partial(self.run_bc, parent), *modifier_args)
            case _:
                raise Exception(f"unknonwn block_type {self}")

        match self.block_immediate:
            case Block.Immediateness.IMMEDIATE:
                return runner(parent, [])
            case Block.Immediateness.DEFERRED:
                return functools.partial(runner, parent)
            case _:
                raise Exception(f"unknonwn block_immediate {self}")

    def run_bc(self, parent: Optional[Frame], args: list):
        if type(self.index) is int:
            return self.vm.bodies[self.index](parent, args)
        elif type(self.index) is list:
            _, _, w, *_ = args
            i = 0 if w is _NOTHING else 1
            if len(self.index[i]) > 1:
                raise NotImplementedError("multiple body blocks") # TODO: all blocks
            return self.vm.bodies[self.index[i][0]](parent, args) 
        else:
            raise Exception("unreachable")

class Stack(list):
    def popn(self, n: int):
        if n == 0:
            return []
        if n > len(self):
            raise IndexError(f"popn n={n} from with len()={len(self)}")
        as_list = self[-n:]
        del self[-n:]
        return as_list

@dataclass
class Body:
    vm: 'VM'
    start: int
    nvars: int
    names: list[int] = field(default_factory=list)
    exported: list[bool] = field(default_factory=list)

    def __call__(self, parent: Optional[Frame], args: list):
        frame = Frame(parent, self.nvars, args)
        pc = self.start
        stack = Stack()
        while True:
            opcode = self.vm.bc[pc]
            print(f"pc: {pc:02d}, op: 0x{opcode:02x}/{opcode}, stack: {stack}")
            match opcode:
                case 0x00: # PUSH
                    arg = self.vm.bc[pc+1]
                    stack.append(self.vm.constants[arg])
                    pc += 2
                case 0x01: # DFND
                    arg = self.vm.bc[pc+1]
                    stack.append(self.vm.blocks[arg](frame))
                    pc += 2
                case 0x06: # POPS
                    _ = stack.pop()
                    pc += 1
                case 0x07: # RETN
                    return stack.pop()
                case 0x0B | 0x0C: # LSTO, LSTM
                    stack.append(stack.popn(self.vm.bc[pc+1]))
                    pc += 2
                case 0x10: # FN1C
                    x, S = stack.popn(2)
                    if x is _NOTHING:
                        raise Exception("𝕩 may not be ·")
                    stack.append(self._call(S, x))
                    pc += 1
                case 0x11: # FN2C
                    x, S, w = stack.popn(3)
                    if w is _NOTHING:
                        raise Exception("𝕨 may not be ·")
                    if x is _NOTHING:
                        raise Exception("𝕨 may not be ·")
                    stack.append(self._call(S, x, w))
                    pc += 1
                case 0x12: # FN1O
                    x, S = stack.popn(2)
                    stack.append(self._call(S, x))
                    pc += 1
                case 0x13: # FN2O
                    x, S, w = stack.popn(3)
                    stack.append(self._call(S, x, w))
                    pc += 1
                case 0x14: # TR2D
                    H, G = stack.popn(2)
                    stack.append(Train2D(G, H))
                    pc += 1
                case 0x15 | 0x17: # TR3D, TR3O
                    H, G, F = stack.popn(3)
                    if F is _NOTHING and opcode == 0x15:
                        raise Exception("𝔽 may not be ·")
                    stack.append(Train3D(G, H, F))
                    pc += 1
                case 0x16: # CHKV
                    raise NotImplementedError
                case 0x1A: # MD1C
                    R, F = stack.popn(2)
                    stack.append(self._call(R, F))
                    pc += 1
                case 0x1B: # MD2C
                    G, R, F = stack.popn(3)
                    stack.append(self._call(R, F, G))
                    pc += 1
                case 0x20: # VARO
                    D, I = self.vm.bc[pc+1:pc+3]
                    stack.append(frame.slot(D, I).value)
                    pc += 3
                case 0x21: # VARM
                    D, I = self.vm.bc[pc+1:pc+3]
                    stack.append(frame.slot(D, I))
                    pc += 3
                case 0x22: # VARU
                    D, I = self.vm.bc[pc+1:pc+3]
                    slot = frame.slot(D, I)
                    stack.append(slot.value)
                    del slot.value
                    pc += 3
                case 0x2C: # NOTM
                    stack.append(_NOTHING) # TODO: Dubious
                    pc += 1
                case 0x30: # SETN
                    val, ref = stack.popn(2)
                    Slot.set(ref, val, allow_uninitialized=True)
                    stack.append(val)
                    pc += 1
                case 0x31: # SETU
                    val, ref = stack.popn(2)
                    Slot.set(ref, val, allow_uninitialized=False)
                    stack.append(val)
                    pc += 1
                case 0x32: # SETM
                    x, F, r = stack.popn(3)
                    Slot.set(r, self._call(F, x, Slot.get(r)))
                    stack.append(Slot.get(r))
                    pc += 1
                case 0x33: # SETC
                    F, r = stack.popn(2)
                    Slot.set(r, self._call(F, Slot.get(r)))
                    stack.append(Slot.get(r))
                    pc += 1
                case _:
                    raise Exception(f"Unknown opcode at {pc}: 0x{opcode:02x}, topstack: {stack[-10:]}")

    @staticmethod
    def _call(F, x = _NOTHING, w = _NOTHING):
        if x is _NOTHING:
            return _NOTHING
        if callable(F):
            return F([F, x, w])
        match F:
            case int() | list() | str():
                return F
            case _:
                raise Exception(f"Unimplemented call type {type(F)} (x={x}, w={w})")

class Array(list):
    shape: list[int]
    fill: Any
    def __init__(self, list: Iterable, shape=None, fill: Any = None):
        super().__init__(list)
        self.shape = shape or [len(self)]
        self.fill = fill

class Provides:
    @staticmethod
    def type(x):
        raise NotImplementedError("type") # https://mlochbaum.github.io/BQN/spec/system.html#operation-properties

    @staticmethod
    def fill(x, w):
        raise NotImplementedError("fill")

    @staticmethod
    def log(x, w):
        raise NotImplementedError("log")

    @staticmethod
    def group_len(x, w):
        raise NotImplementedError("log")

    @staticmethod
    def group_ord(x, w):
        raise NotImplementedError("log")

    @staticmethod
    def assert_fn(x, w):
        raise NotImplementedError("assert_fn")

    @staticmethod
    def plus(x, w):
        raise NotImplementedError("plus")

    @staticmethod
    def minus(x, w):
        raise NotImplementedError("minus")

    @staticmethod
    def times(x, w):
        raise NotImplementedError("times")

    @staticmethod
    def divide(x, w):
        raise NotImplementedError("divide")

    @staticmethod
    def power(x, w):
        raise NotImplementedError("power")

    @staticmethod
    def floor(x, w):
        raise NotImplementedError("floor")

    @staticmethod
    def equals(x, w):
        raise NotImplementedError("equals")

    @staticmethod
    def leqt(x, w):
        raise NotImplementedError("leqt")

    @staticmethod
    def shape(x, w):
        raise NotImplementedError("shape")

    @staticmethod
    def deshape(args):
        _, x, w, *_ = args
        raise NotImplementedError("deshape")

    @staticmethod
    def pick(x, w):
        raise NotImplementedError("pick")

    @staticmethod
    def range(x, w):
        raise NotImplementedError("range")

    @staticmethod
    def table(args):
        _, f, _ = args
        def inner_table(args):
            return Array([f([f, x, w]) for (x, w) in itertools.product(args[1], args[2], )], args[2].shape + args[1].shape)  if args[2] is not _NOTHING else Array([f([f, x, _NOTHING]) for x in args[1]], args[1].shape)
        return inner_table

    @staticmethod
    def scan(args):
        _, f, _ = args
        def inner_scan(args):
            _, x, w, *_ = args
            if not x or x.shape == [0]:
                raise ValueError("`: 𝕩 must have rank at least 1")
            if w is not _NOTHING:
                raise NotImplementedError("scan with 𝕨")
            else:
                raise NotImplementedError("f")
        return MD1(inner_scan, *args)

    @staticmethod
    def fill_by(args):
        _, f, g = args
        def inner_fill(args):
            _, x, w, *_ = args
            r = f([g, x, w])
            return r

        return MD2(inner_fill, *args)  # https://mlochbaum.github.io/BQN/implementation/vm.html#testing

    @staticmethod
    def valences(args):
        _, f, g = args
        def inner_valences(args):
            _, x, w, *_ = args
            if w is not _NOTHING:
                return g([g, x, w])
            else:
                return f([f, x, w])
        return MD2(inner_valences, *args)

    @staticmethod
    def catches(x, w):
        raise NotImplementedError("catches")

    def __class_getitem__(cls, key):
        provides = [
            cls.type, cls.fill, cls.log, cls.group_len, cls.group_ord,
            cls.assert_fn, cls.plus, cls.minus, cls.times, cls.divide,
            cls.power, cls.floor, cls.equals, cls.leqt, cls.shape,
            cls.deshape, cls.pick, cls.range, cls.table, cls.scan,
            cls.fill_by, cls.valences, cls.catches,
        ]
        return provides[key]

class VM:
    def __init__(self, bc: list[int], constants: list[Any], blocks, bodies) -> None:
        self.bc = bc
        self.constants = constants
        self.bodies = [Body(self, *x) for x in bodies]
        self.blocks = [Block(self, *x) for x in blocks]

    def __call__(self):
        return self.blocks[0](None)

if __name__ == "__main__":
    input = [[0,0,0,1,0,2,0,3,0,2,1,1,11,2,11,2,11,2,11,2,11,2,1,2,0,0,0,0,1,3,11,2,11,2,26,16,7,34,0,1,7,34,0,1,1,4,16,7,34,0,1,7,34,0,1,33,0,3,33,0,4,12,2,48,6,34,0,0,33,0,5,48,6,0,0,1,5,1,6,26,16,6,1,7,32,0,4,32,0,5,16,26,7,34,0,1,32,1,4,16,7,34,0,0,6,1,8,33,1,5,49,7,34,0,1,33,0,5,33,0,6,12,2,48,6,34,0,6,34,0,4,16,7,34,0,0,6,1,9,7,34,0,1,33,0,3,33,0,4,12,2,48,6,34,0,3,7],[0,1,4,3],[[0,1,0],[0,0,1],[1,1,2],[0,0,3],[0,0,4],[1,1,5],[0,0,6],[1,0,7],[0,0,8],[0,0,9]],[[0,0],[37,3],[41,2],[48,3],[52,6],[93,2],[101,3],[112,7],[133,3],[140,5]]]
    print(f"""bc:        {input[0]}
constants: {input[1]},
blocks:    {input[2]},
bodies:    {input[3]}""")
    print(VM()())
