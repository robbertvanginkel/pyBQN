import unittest

from pybqn.provide import provides
from pybqn.vm import VM
from pybqn.runtime import r0, r1, r


class RuntimeTest(unittest.TestCase):
    def test_r0(self):
        vm = VM(*r0(provides))
        self.assertIsNotNone(vm())

    def test_r1(self):
        runtime_0 = VM(*r0(provides))()
        runtime_1 = VM(*r1(provides, runtime_0))()
        self.assertIsNotNone(runtime_1)

    def test_runtime(self):
        self.assertIsNotNone(VM(*r(provides))())


if __name__ == "__main__":
    unittest.main()
