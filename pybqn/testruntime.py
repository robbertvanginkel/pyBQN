import unittest

from pybqn.vm import Provides, VM
from pybqn.runtime import r0, r1, r


class RuntimeTest(unittest.TestCase):
    def test_r0(self):
        vm = VM(*r0(Provides))
        self.assertIsNotNone(vm())

    def test_r1(self):
        runtime_0 = VM(*r0(Provides))()
        runtime_1 = VM(*r1(Provides, runtime_0))()
        self.assertIsNotNone(runtime_1)

    def test_runtime(self):
        self.assertIsNotNone(VM(*r(Provides))())


if __name__ == "__main__":
    unittest.main()
