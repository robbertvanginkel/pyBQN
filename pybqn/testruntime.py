import inspect
from multiprocessing import Value
import unittest
import math

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


class PrimitiveTests(unittest.TestCase):
    """https://github.com/mlochbaum/BQN/blob/f4b09b675386c789c8296c96874871916a3abdcf/test/cases/prim.bqn

    pbpaste | tr '\n' '\0'| xargs -n1 -0 bqn utils/cpy.bqn ../../BQN
    """
    def test_layer0(self):
        runtime, set_prims, set_inv = VM(*r(provides))()
        # fmt: off
        cases = {
            "0≡¯2+2"		    : (1, [[0,4,0,0,0,3,17,0,1,0,2,17,7],[runtime[0],runtime[18],0,-2,2],[[0,1,0]],[[0,0]]]),
            "1e4≡5e3+5e3"       : (1, [[0,3,0,0,0,3,17,0,1,0,2,17,7],[runtime[0],runtime[18],10000,5000],[[0,1,0]],[[0,0]]]),
            "'c'≡'a'+2"		    : (1, [[0,2,0,0,0,4,17,0,1,0,3,17,7],[runtime[0],runtime[18],2,'c','a'],[[0,1,0]],[[0,0]]]),
            "'a'≡¯2+'c'"	    : (1, [[0,4,0,0,0,2,17,0,1,0,3,17,7],[runtime[0],runtime[18],-2,'a','c'],[[0,1,0]],[[0,0]]]),
            "'a'+'c'"           : (TypeError, [[0,2,0,0,0,1,17,7],[runtime[0],'a','c'],[[0,1,0]],[[0,0]]]),
            "F←-⋄f+2"           : (TypeError, [[0,1,33,0,0,48,6,0,2,0,0,34,0,0,17,7],[runtime[0],runtime[1],2],[[0,1,0]],[[0,1]]]),
            "¯4≡+¯4"            : (1, [[0,2,0,0,16,0,1,0,2,17,7],[runtime[0],runtime[18],-4],[[0,1,0]],[[0,0]]]),
            "+'x'"              : (TypeError, [[0,1,0,0,16,7],[runtime[0],'x'],[[0,1,0]],[[0,0]]]),
            "F←-⋄+f"            : (TypeError, [[0,1,33,0,0,48,6,34,0,0,0,0,16,7],[runtime[0],runtime[1]],[[0,1,0]],[[0,1]]]),
            "¯∞≡1e6-∞"          : (1, [[0,4,0,0,0,3,17,0,1,0,2,17,7],[runtime[1],runtime[18],-math.inf,1000000,math.inf],[[0,1,0]],[[0,0]]]),
            "4≡-¯4"             : (1, [[0,3,0,0,16,0,1,0,2,17,7],[runtime[1],runtime[18],4,-4],[[0,1,0]],[[0,0]]]),
            "¯∞≡-∞"             : (1, [[0,3,0,0,16,0,1,0,2,17,7],[runtime[1],runtime[18],-math.inf,math.inf],[[0,1,0]],[[0,0]]]),
            "∞≡-¯∞"             : (1, [[0,3,0,0,16,0,1,0,2,17,7],[runtime[1],runtime[18],math.inf,-math.inf],[[0,1,0]],[[0,0]]]),
            "4≡9-5"             : (1, [[0,4,0,0,0,3,17,0,1,0,2,17,7],[runtime[1],runtime[18],4,9,5],[[0,1,0]],[[0,0]]]),
            "@≡'a'-97"          : (1, [[0,2,0,0,0,4,17,0,1,0,3,17,7],[runtime[1],runtime[18],97,'\0','a'],[[0,1,0]],[[0,0]]]),
            "3≡'d'-'a'"         : (1, [[0,4,0,0,0,3,17,0,1,0,2,17,7],[runtime[1],runtime[18],3,'d','a'],[[0,1,0]],[[0,0]]]),
            "'Q'≡'q'+'A'-'a'"   : (1, [[0,6,0,1,0,5,17,0,0,0,4,17,0,2,0,3,17,7],[runtime[0],runtime[1],runtime[18],'Q','q','A','a'],[[0,1,0]],[[0,0]]]),
            "97-'a'"            : (TypeError, [[0,2,0,0,0,1,17,7],[runtime[1],97,'a'],[[0,1,0]],[[0,0]]]),
            "@-1"               : (ValueError, [[0,1,0,0,0,2,17,7],[runtime[1],1,'\0'],[[0,1,0]],[[0,0]]]),
            "-'a'"              : (TypeError, [[0,1,0,0,16,7],[runtime[1],'a'],[[0,1,0]],[[0,0]]])
        }
        # fmt: on
        for case, (expected, input) in cases.items():
            with self.subTest(case=case):
                if inspect.isclass(expected) and issubclass(expected, Exception):
                    with self.assertRaises(expected):
                        self.assertEqual(VM(*input)(), expected)
                else:
                    self.assertEqual(VM(*input)(), expected)


if __name__ == "__main__":
    unittest.main()
