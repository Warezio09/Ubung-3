import unittest
import numpy as np
from ode import forward_difference_quotient, backward_difference_quotient, central_difference_quotient, explicit_euler, implicit_euler


class TestODE(unittest.TestCase):
    def test_DQ_value_error(self):
        with self.subTest("Test forward DQ value error"):
            with self.assertRaises(ValueError):
                forward_difference_quotient(np.sin, 0, -1)
            with self.assertRaises(ValueError):
                forward_difference_quotient(np.sin, 0, 0)


if __name__ == '__main__':
    unittest.main()
