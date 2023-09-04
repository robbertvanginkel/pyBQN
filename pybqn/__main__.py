from pybqn.provide import make_prims, provides, ptype, fmt_num
from pybqn.runtime import bqnstr, r, c, f
from pybqn.vm import VM, Array, call

if __name__ == "__main__":
    runtime, set_prims, _ = VM(*r(provides))()
    decompose, prim_ind, glyph = make_prims(runtime)
    call(set_prims, Array([decompose, prim_ind]))

    compiler = VM(*c(runtime))()
    compiled = call(compiler, x=bqnstr('<⟜\'a\'⊸/ "Big Questions Notation"'), w=runtime)

    formatter = VM(*f(runtime))()
    fmt, _ = call(formatter, Array([ptype, formatter, glyph, fmt_num]))

    result = VM(*compiled[:4])()
    print("".join(call(fmt, result)))
