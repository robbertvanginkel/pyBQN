import math
import unittest

from pybqn.program import Array, BQNError, Program, call
from pybqn.provide import bqnstr, make_prims, provides
from pybqn.vm import VM


class VMTest(unittest.TestCase):
    def test_pi(self):
        """
        somewhere in c.bqn there's parsing of numbers, which happens with a scan. When the π literal is
        used, it's value in the list will be a float which causes all operations after to also become 
        floats unless the float*0 turns into an int. Validate here we catch this case.
        """
        vm = VM()
        compiled = vm.compile(bqnstr("⟨π, 1⟩"))
        self.assertEqual(
            list(map(type, compiled.constants)), [type(math.pi), type(int(1))]
        )
        result = compiled()
        self.assertEqual(result, [math.pi, 1])
        self.assertEqual(vm.compile(bqnstr("⟨π, ↕1⟩"))(), [math.pi, [0]])
