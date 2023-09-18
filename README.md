# PyBQN: a [BQN](https://mlochbaum.github.io/BQN/) implementation in python
A(n incomplete) BQN [VM](https://mlochbaum.github.io/BQN/implementation/vm.html) implementation in python.

    $ python3 -m pybqn
    bqn> <⟜'a'⊸/ "Big Questions Notation"
    "B Q N"

This is a toy interpreter to better understand the language and VM. If you're looking for more performant BQN from python consider the [BQN↔NumPy bridge](https://github.com/vmchale/pybqn) or writing your own interface to [CBQN](https://github.com/dzaima/CBQN).

## Status
- Only implements the bare minimum [core runtime](https://github.com/mlochbaum/BQN) and is _slow_.
- Missing some opcodes/features: namespaces, headers, multi-body blocks.
- Only system funcs to run [`test/this.bqn`](https://github.com/mlochbaum/BQN/tree/master/test) tests, and path/wdpath handling is not up to spec.
- Only REP~~L~~ (without the L), missing redefinable variables and passing them into the compile (https://mlochbaum.github.io/BQN/implementation/vm.html#compiler-arguments).
- Passing some tests
  - Pass: bytecode, identity, literal, prim, simple, token, under
  - Fail: fill, header, namespace, syntax, undo, unhead

# License
The python code is licensed as MIT. Some unittests used to validate the VM/runtime before it could host the compiler contain pre-compiled snippets from the https://github.com/mlochbaum/BQN repo, see the respective license there.