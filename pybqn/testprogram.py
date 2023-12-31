import unittest

from pybqn.program import Program


class ByteCodeTest(unittest.TestCase):
    """https://github.com/mlochbaum/BQN/blob/master/test/cases/bytecode.bqn"""

    # fmt: off
    cases = {
        "5"                        : (5, [[0,0,7],[5],[[0,1,0]],[[0,0]]]),
        "4⋄3"                      : (3, [[0,0,6,0,1,7],[4,3],[[0,1,0]],[[0,0]]]),
        "a←5"                      : (5, [[0,0,33,0,0,48,7],[5],[[0,1,0]],[[0,1]]]),
        "a←5⋄a↩4"                  : (4, [[0,0,33,0,0,48,6,0,1,33,0,0,49,7],[5,4],[[0,1,0]],[[0,1]]]),
        "a←2⋄b←3⋄a"                : (2, [[0,0,33,0,0,48,6,0,1,33,0,1,48,6,34,0,0,7],[2,3],[[0,1,0]],[[0,2]]]),
        "a←1⋄A 4"                  : (1, [[0,0,33,0,0,48,6,34,0,0,7],[1],[[0,1,0]],[[0,1]]]),
        "a←2⋄3 A 4"                : (2, [[0,0,33,0,0,48,6,0,2,34,0,0,0,1,17,7],[2,3,4],[[0,1,0]],[[0,1]]]),
        "{𝕩}6"                     : (6, [[0,0,1,1,16,7,34,0,1,7],[6],[[0,1,0],[0,0,1]],[[0,0],[6,3]]]),
        "A←{𝕨}⋄3 A 4"              : (3, [[1,1,33,0,0,48,6,0,1,34,0,0,0,0,17,7,34,0,2,7],[3,4],[[0,1,0],[0,0,[[],[1]]]],[[0,1],[16,3]]]),
        "a‿b←7‿2⋄a"                : (7, [[0,0,0,1,11,2,33,0,0,33,0,1,12,2,48,6,34,0,0,7],[7,2],[[0,1,0]],[[0,2]]]),
        "·‿b←7‿2⋄b"                : (2, [[0,0,0,1,11,2,44,33,0,0,12,2,48,6,34,0,0,7],[7,2],[[0,1,0]],[[0,1]]]),
        "0{𝕨𝕏1}2"                  : (2, [[0,2,1,1,0,0,17,7,0,1,34,0,1,34,0,2,19,7],[0,1,2],[[0,1,0],[0,0,1]],[[0,0],[8,3]]]),
        "{({𝕨}𝕨)𝕏𝕩}5"              : (5, [[0,0,1,1,16,7,32,0,1,34,0,1,34,0,2,1,2,18,19,7,34,0,2,7],[5],[[0,1,0],[0,0,1],[0,0,[[],[2]]]],[[0,0],[6,3],[20,3]]]),
        "{𝕩{a‿b←𝕨}𝕨,𝕩}8"           : (8, [[0,0,1,1,16,7,34,0,2,1,2,32,0,1,19,6,34,0,1,7,34,0,2,33,0,3,33,0,4,12,2,48,7],[8],[[0,1,0],[0,0,1],[0,0,[[],[2]]]],[[0,0],[6,3],[20,5]]]),
        "4{𝔽}"                     : (4, [[1,1,0,0,26,7,34,0,1,7],[4],[[0,1,0],[1,1,1]],[[0,0],[6,2]]]),
        "4{𝔽⋄𝕩}6"                  : (6, [[0,1,1,1,0,0,26,16,7,34,0,4,6,34,0,1,7],[4,6],[[0,1,0],[1,0,1]],[[0,0],[9,5]]]),
        "3{𝔾}{𝕩} 1"                : (1, [[0,1,1,1,1,2,0,0,27,16,7,34,0,1,7,34,0,2,7],[3,1],[[0,1,0],[0,0,1],[2,1,2]],[[0,0],[11,3],[15,3]]]),
        "(2{𝔽}{𝕩})3"               : (2, [[0,1,1,1,1,2,0,0,26,20,16,7,34,0,1,7,34,0,1,7],[2,3],[[0,1,0],[0,0,1],[1,1,2]],[[0,0],[12,3],[16,2]]]),
        "3({a‿b←𝕩⋄a}{𝕨‿𝕩})4"       : (3, [[0,1,1,1,1,2,20,0,0,17,7,34,0,2,34,0,1,11,2,7,34,0,1,33,0,3,33,0,4,12,2,48,6,34,0,3,7],[3,4],[[0,1,0],[0,0,[[],[1]]],[0,0,2]],[[0,0],[11,3],[20,5]]]),
        "4({𝕨‿𝕩}{𝕩}{𝕨})5"          : (4, [[0,1,1,1,1,2,1,3,21,0,0,17,7,34,0,2,7,34,0,1,7,34,0,2,34,0,1,11,2,7],[4,5],[[0,1,0],[0,0,[[],[1]]],[0,0,2],[0,0,[[],[3]]]],[[0,0],[13,3],[17,3],[21,3]]]),
        "a‿b←(2{𝕨‿𝕩}{𝕩})5⋄a"       : (2, [[0,1,1,1,1,2,0,0,21,16,33,0,0,33,0,1,12,2,48,6,34,0,0,7,34,0,1,7,34,0,2,34,0,1,11,2,7],[2,5],[[0,1,0],[0,0,1],[0,0,[[],[2]]]],[[0,2],[24,3],[28,3]]]),
        "({a↩2⋄𝕩}{𝕩⋄a}{a↩3⋄𝕩})a←4" : (2, [[0,2,33,0,0,48,1,1,1,2,1,3,21,16,7,0,1,33,1,0,49,6,34,0,1,7,34,0,1,6,32,1,0,7,0,0,33,1,0,49,6,34,0,1,7],[2,3,4],[[0,1,0],[0,0,1],[0,0,2],[0,0,3]],[[0,1],[15,3],[26,3],[34,3]]]),
        "a←3⋄a{𝕩}↩8⋄a"             : (8, [[0,0,33,0,0,48,6,0,1,1,1,33,0,0,50,6,34,0,0,7,34,0,1,7],[3,8],[[0,1,0],[0,0,1]],[[0,1],[20,3]]]),
        "a←4⋄a{𝕨⋄5}↩6"             : (5, [[0,0,33,0,0,48,6,0,2,1,1,33,0,0,50,7,34,0,2,6,0,1,7],[4,5,6],[[0,1,0],[0,0,1]],[[0,1],[16,3]]]),
        "a←3⋄a{𝕩⋄1}↩⋄a"            : (1, [[0,0,33,0,0,48,6,1,1,33,0,0,51,6,34,0,0,7,34,0,1,6,0,1,7],[3,1],[[0,1,0],[0,0,1]],[[0,1],[18,3]]]),
        "a‿b←2‿1⋄a‿b{𝕩‿𝕨}↩4⋄a"     : (4, [[0,0,0,1,11,2,33,0,0,33,0,1,12,2,48,6,0,2,1,1,33,0,0,33,0,1,12,2,50,6,34,0,0,7,34,0,1,34,0,2,11,2,7],[2,1,4],[[0,1,0],[0,0,[[],[1]]]],[[0,2],[34,3]]]),

        "{𝕨{a←𝕩⋄{a↩𝕩}𝕨⋄a}𝕩}7"      : (7, [[0,0,1,1,16,7,34,0,1,1,2,34,0,2,19,7,34,0,1,33,0,3,48,6,34,0,2,1,3,18,6,32,0,3,7,34,0,1,33,1,3,49,7],[7],[[0,1,0],[0,0,1],[0,0,2],[0,0,3]],[[0,0],[6,3],[16,4],[35,3]]]),
        "3{𝕨{a←𝕩⋄{a↩𝕩}𝕨⋄a}𝕩}7"     : (3, [[0,1,1,1,0,0,17,7,34,0,1,1,2,34,0,2,19,7,34,0,1,33,0,3,48,6,34,0,2,1,3,18,6,32,0,3,7,34,0,1,33,1,3,49,7],[3,7],[[0,1,0],[0,0,1],[0,0,2],[0,0,3]],[[0,0],[8,3],[18,4],[37,3]]]),
        "{𝕏0} {𝕨{a←𝕩⋄{a↩𝕩}𝕨⋄a}𝕏}7" : (7, [[0,1,1,1,16,1,2,16,7,34,0,1,1,3,34,0,2,23,7,0,0,34,0,1,16,7,34,0,1,33,0,3,48,6,34,0,2,1,4,18,6,32,0,3,7,34,0,1,33,1,3,49,7],[0,7],[[0,1,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4]],[[0,0],[9,3],[19,3],[26,4],[45,3]]]),
        "{𝕏0}3{𝕨{a←𝕩⋄{a↩𝕩}𝕨⋄a}𝕏}7" : (3, [[0,2,1,1,0,1,17,1,2,16,7,34,0,1,1,3,34,0,2,23,7,0,0,34,0,1,16,7,34,0,1,33,0,3,48,6,34,0,2,1,4,18,6,32,0,3,7,34,0,1,33,1,3,49,7],[0,3,7],[[0,1,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4]],[[0,0],[11,3],[21,3],[28,4],[47,3]]]),
        
        "a←1⋄{a←2}⋄a"                   : (1, [[0,0,33,0,0,48,6,1,1,6,34,0,0,7,0,1,33,0,0,48,7],[1,2],[[0,1,0],[0,1,1]],[[0,1],[14,1]]]),
        "a←1⋄{a↩2}⋄a"                   : (2, [[0,0,33,0,0,48,6,1,1,6,32,0,0,7,0,1,33,1,0,49,7],[1,2],[[0,1,0],[0,1,1]],[[0,1],[14,0]]]),
        "f‿g←{a←2⋄{a↩𝕩}‿{𝕩⋄a}}⋄F 6⋄G 0" : (6, [[1,1,33,0,0,33,0,1,12,2,48,6,0,1,34,0,0,16,6,0,2,34,0,1,16,7,0,0,33,0,0,48,6,1,2,1,3,11,2,7,34,0,1,33,1,0,49,7,34,0,1,6,32,1,0,7],[2,6,0],[[0,1,0],[0,1,1],[0,0,2],[0,0,3]],[[0,2],[26,1],[40,3],[48,3]]]),
        "L←{𝕩{𝕏𝕗}}⋄{𝕏𝕤}L L L 5"         : (5, [[1,1,33,0,0,48,6,0,0,32,0,0,16,32,0,0,16,34,0,0,16,1,2,16,7,1,3,34,0,1,26,7,34,0,0,34,0,1,16,7,34,0,4,34,0,1,16,7],[5],[[0,1,0],[0,0,1],[0,0,2],[1,0,3]],[[0,1],[25,3],[32,3],[40,5]]]),
        "_l←{𝕩{𝕏𝕗} 𝔽}⋄{𝕏𝕤} {𝕩}_l 3 _l 5": (3, [[1,1,33,0,0,48,6,0,1,32,0,0,0,0,26,16,34,0,0,1,2,26,16,1,3,16,7,34,0,4,1,4,34,0,1,26,20,7,34,0,1,7,34,0,0,34,0,1,16,7,34,0,4,34,0,1,16,7],[3,5],[[0,1,0],[1,0,1],[0,0,2],[0,0,3],[1,0,4]],[[0,1],[27,5],[38,3],[42,3],[50,5]]]),
        "1{𝕨}{𝔽{𝕩𝔽𝕨}𝔾𝔽}{𝕩}0"            : (1, [[0,1,1,1,1,2,1,3,27,0,0,17,7,34,0,1,7,32,0,1,34,0,2,1,4,34,0,1,26,21,7,34,0,2,7,34,0,2,34,0,4,34,0,1,17,7],[1,0],[[0,1,0],[0,0,1],[2,1,2],[0,0,[[],[3]]],[1,0,[[],[4]]]],[[0,0],[13,3],[17,3],[31,3],[35,5]]]),
        "0‿(0‿{𝕩}){{a‿b←𝕩⋄t←𝕤⋄{𝕤⋄T↩{𝕤⋄{a‿b←𝕩⋄a}}}{B𝕗}0⋄(T b){a‿b←𝕩⋄𝔽b}}𝕗} 0‿(1‿(2‿(3‿(4‿{𝕩}))))": (2, [[0,0,0,1,0,2,0,3,0,4,1,1,11,2,11,2,11,2,11,2,11,2,1,2,0,0,0,0,1,3,11,2,11,2,26,16,7,34,0,1,7,34,0,1,1,4,16,7,34,0,1,7,34,0,1,33,0,3,33,0,4,12,2,48,6,34,0,0,33,0,5,48,6,0,0,1,5,1,6,26,16,6,1,7,32,0,4,32,0,5,16,26,7,34,0,1,32,1,4,16,7,34,0,0,6,1,8,33,1,5,49,7,34,0,1,33,0,5,33,0,6,12,2,48,6,34,0,6,34,0,4,16,7,34,0,0,6,1,9,7,34,0,1,33,0,3,33,0,4,12,2,48,6,34,0,3,7],[0,1,2,3,4],[[0,1,0],[0,0,1],[1,1,2],[0,0,3],[0,0,4],[1,1,5],[0,0,6],[1,0,7],[0,0,8],[0,0,9]],[[0,0],[37,3],[41,2],[48,3],[52,6],[93,2],[101,3],[112,7],[133,3],[140,5]]]),
    }
    # fmt: on

    def test_program(self):
        for case, (expected, input) in self.cases.items():
            with self.subTest(case=case):
                self.assertEqual(Program(*input)(), expected)
