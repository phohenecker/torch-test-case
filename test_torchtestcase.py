#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import operator
import unittest

import torch

from unittest import mock

from torch import nn
from torch.nn.utils import rnn

import torchtestcase as ttc


__author__ = "Patrick Hohenecker"
__copyright__ = (
        "Copyright (c) 2018 Patrick Hohenecker\n"
        "\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        "of this software and associated documentation files (the \"Software\"), to deal\n"
        "in the Software without restriction, including without limitation the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
        "copies of the Software, and to permit persons to whom the Software is\n"
        "furnished to do so, subject to the following conditions:\n"
        "\n"
        "The above copyright notice and this permission notice shall be included in all\n"
        "copies or substantial portions of the Software.\n"
        "\n"
        "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
        "SOFTWARE."
)
__license__ = "MIT License"
__version__ = "2018.1"
__date__ = "Aug 18, 2018"
__maintainer__ = "Patrick Hohenecker"
__email__ = "mail@paho.at"
__status__ = "Development"


class TorchTestCaseTest(unittest.TestCase):
    
    def setUp(self):
        self.test_case = ttc.TorchTestCase()

    # noinspection PyArgumentList
    def test_assert_packed_sequence_equal(self):
        float_seq_1 = rnn.pack_padded_sequence(
                torch.FloatTensor([[1, 1], [1, 0]]),
                [2, 1]
        )
        float_seq_2 = rnn.pack_padded_sequence(
                torch.FloatTensor([[1, 1], [1, 0]]),
                [2, 2]
        )
        float_seq_3 = rnn.pack_padded_sequence(
                torch.FloatTensor([[1, 1], [1, 1], [1, 0]]),
                [3, 2]
        )
        float_seq_4 = rnn.pack_padded_sequence(
                torch.FloatTensor([[1, 1, 1], [1, 0, 0]]),
                [2, 1, 1]
        )
        long_seq = rnn.pack_padded_sequence(
                torch.LongTensor([[1, 1], [1, 0]]),
                [2, 1]
        )

        # CHECK: the assertion fails if either of the args is not a sequence
        with self.assertRaises(AssertionError):
            self.test_case.assert_packed_sequence_equal("no sequence", float_seq_1)
        with self.assertRaises(AssertionError):
            self.test_case.assert_packed_sequence_equal(float_seq_1, torch.zeros(3))

        # CHECK: the assertion fails if the args contain data of different types
        with self.assertRaises(AssertionError):
            self.test_case.assert_packed_sequence_equal(float_seq_1, long_seq)

        # CHECK: the assertion fails if the args differ in values or shape
        with self.assertRaises(AssertionError):
            self.test_case.assert_packed_sequence_equal(float_seq_1, float_seq_2)
        with self.assertRaises(AssertionError):
            self.test_case.assert_packed_sequence_equal(float_seq_1, float_seq_3)
        with self.assertRaises(AssertionError):
            self.test_case.assert_packed_sequence_equal(float_seq_1, float_seq_4)

        # CHECK: no errors are raised for equal sequences
        self.test_case.assert_packed_sequence_equal(
                rnn.pack_padded_sequence(torch.FloatTensor([[1, 1], [1, 0]]), [2, 1]),
                rnn.pack_padded_sequence(torch.FloatTensor([[1, 1], [1, 0]]), [2, 1])
        )

    # noinspection PyArgumentList
    def test_assert_parameter_equal(self):
        # CHECK: the assertion fails if either of the args is not a parameter
        with self.assertRaises(AssertionError):
            self.test_case.assert_parameter_equal("no tensor", nn.Parameter(torch.zeros(3)))
        with self.assertRaises(AssertionError):
            self.test_case.assert_parameter_equal(nn.Parameter(torch.zeros(3)), torch.zeros(3))
    
        # CHECK: the assertion fails if the args contain data of different types
        with self.assertRaises(AssertionError):
            self.test_case.assert_parameter_equal(
                    nn.Parameter(torch.FloatTensor([0])),
                    nn.Parameter(torch.LongTensor([0]), requires_grad=False)
            )
    
        # CHECK: the assertion fails if the args differ in values or shape
        with self.assertRaises(AssertionError):
            self.test_case.assert_parameter_equal(
                    nn.Parameter(torch.FloatTensor([0, 1])),
                    nn.Parameter(torch.FloatTensor([0, 2]))
            )
        with self.assertRaises(AssertionError):
            self.test_case.assert_parameter_equal(
                    nn.Parameter(torch.FloatTensor([0, 1])),
                    nn.Parameter(torch.FloatTensor([0, 1, 2]))
            )
        with self.assertRaises(AssertionError):
            self.test_case.assert_parameter_equal(
                    nn.Parameter(torch.FloatTensor([0, 1])),
                    nn.Parameter(torch.FloatTensor([[0, 1]]))
            )
    
        # CHECK: no errors are raised for equal parameters
        self.test_case.assert_parameter_equal(
                nn.Parameter(torch.zeros(3)),
                nn.Parameter(torch.zeros(3))
        )

    # noinspection PyArgumentList
    def test_assert_tensor_equal(self):
        # CHECK: the assertion fails if either of the args is not a tensor
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_equal("no tensor", torch.zeros(3))
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_equal(torch.zeros(3), 666)
        
        # CHECK: the assertion fails if the args are tensors of different types
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_equal(torch.FloatTensor([0]), torch.LongTensor([0]))
        
        # CHECK: the assertion fails if the args differ in values or shape
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_equal(
                    torch.FloatTensor([0, 1]),
                    torch.FloatTensor([0, 2])
            )
        self.test_case.eps = 1e-9  # <<<<<<<<<< change of tolerance
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_equal(
                    torch.zeros(3),
                    11e-10 * torch.ones(3)
            )
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_equal(
                    torch.FloatTensor([0, 1]),
                    torch.FloatTensor([0, 1, 2])
            )
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_equal(
                    torch.FloatTensor([0, 1]),
                    torch.FloatTensor([[0, 1]])
            )
        
        # CHECK: no errors are raised for (approximately) equal tensors
        self.test_case.eps = 0
        self.test_case.assert_tensor_equal(torch.zeros(3), torch.zeros(3))
        self.test_case.eps = 1e-9
        self.test_case.assert_tensor_equal(torch.zeros(3), 1e-9 * torch.ones(3))

    # noinspection PyArgumentList
    def test_assert_tensor_greater(self):
        # CHECK: no error is raised if the assertion is True
        self.test_case.assert_tensor_greater(torch.ones(3), torch.zeros(3))
        self.test_case.assert_tensor_greater(1, torch.zeros(3))
        self.test_case.assert_tensor_greater(torch.ones(3), 0)
        
        # CHECK: the assertion fails if the comparison does not apply
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_greater(
                    torch.FloatTensor([0, 1]),
                    torch.FloatTensor([1, 0])
            )
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_greater(torch.FloatTensor([0, 1]), 1)
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_greater(1, torch.FloatTensor([0, 1]))

    # noinspection PyArgumentList
    def test_assert_tensor_greater_equal(self):
        # CHECK: no error is raised if the assertion is True
        self.test_case.assert_tensor_greater_equal(torch.ones(3), torch.ones(3))
        self.test_case.assert_tensor_greater_equal(torch.ones(3), torch.zeros(3))
        self.test_case.assert_tensor_greater_equal(1, torch.zeros(3))
        self.test_case.assert_tensor_greater_equal(torch.ones(3), 1)
    
        # CHECK: the assertion fails if the comparison does not apply
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_greater_equal(
                    torch.FloatTensor([0, 1]),
                    torch.FloatTensor([1, 0])
            )
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_greater_equal(torch.FloatTensor([0, 1]), 2)
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_greater_equal(0, torch.FloatTensor([0, 1]))

    # noinspection PyArgumentList
    def test_assert_tensor_less(self):
        # CHECK: no error is raised if the assertion is True
        self.test_case.assert_tensor_less(torch.zeros(3), torch.ones(3))
        self.test_case.assert_tensor_less(0, torch.ones(3))
        self.test_case.assert_tensor_less(torch.zeros(3), 1)
    
        # CHECK: the assertion fails if the comparison does not apply
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_less(
                    torch.FloatTensor([0, 1]),
                    torch.FloatTensor([1, 0])
            )
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_less(torch.FloatTensor([0, 1]), 1)
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_less(0, torch.FloatTensor([0, 1]))

    # noinspection PyArgumentList
    def test_assert_tensor_less_equal(self):
        # CHECK: no error is raised if the assertion is True
        self.test_case.assert_tensor_less_equal(torch.ones(3), torch.ones(3))
        self.test_case.assert_tensor_less_equal(torch.zeros(3), torch.ones(3))
        self.test_case.assert_tensor_less_equal(0, torch.ones(3))
        self.test_case.assert_tensor_less_equal(torch.zeros(3), 1)
    
        # CHECK: the assertion fails if the comparison does not apply
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_less_equal(
                    torch.FloatTensor([0, 1]),
                    torch.FloatTensor([1, 0])
            )
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_less_equal(torch.FloatTensor([0, 1]), 0)
        with self.assertRaises(AssertionError):
            self.test_case.assert_tensor_less_equal(2, torch.FloatTensor([0, 1]))

    # noinspection PyArgumentList
    @staticmethod
    def test_assertEqual():
        # CHECK: assert_packed_sequence_equal is invoked appropriately
        with mock.patch.object(ttc.TorchTestCase, "assert_packed_sequence_equal") as mock_method:
            ttc.TorchTestCase().assertEqual(
                    rnn.pack_padded_sequence(torch.FloatTensor([[1, 1], [1, 0]]), [2, 1]),
                    rnn.pack_padded_sequence(torch.FloatTensor([[1, 1], [1, 0]]), [2, 1])
            )
        mock_method.assert_called_once()
    
        # CHECK: assert_parameter_equal is invoked appropriately
        with mock.patch.object(ttc.TorchTestCase, "assert_parameter_equal") as mock_method:
            ttc.TorchTestCase().assertEqual(nn.Parameter(torch.zeros(3)), nn.Parameter(torch.zeros(3)))
        mock_method.assert_called_once()
        
        # CHECK: assert_tensor_equal is invoked appropriately
        with mock.patch.object(ttc.TorchTestCase, "assert_tensor_equal") as mock_method:
            ttc.TorchTestCase().assertEqual(torch.zeros(3), torch.zeros(3))
        mock_method.assert_called_once()
    
    @staticmethod
    def test_assertGreater():
        # CHECK: the original assertGreater method is invoked appropriately
        with mock.patch.object(ttc.TorchTestCase, "assertGreater") as mock_method:
            ttc.TorchTestCase().assertGreater(1, 0)
        mock_method.assert_called_once()

        # CHECK: assert_tensor_greater is invoked appropriately
        with mock.patch.object(ttc.TorchTestCase, "assert_tensor_greater") as mock_method:
            ttc.TorchTestCase().assertGreater(1, torch.zeros(3))
        mock_method.assert_called_once()

    @staticmethod
    def test_assertGreaterEqual():
        # CHECK: the original assertGreaterEqual method is invoked appropriately
        with mock.patch.object(ttc.TorchTestCase, "assertGreaterEqual") as mock_method:
            ttc.TorchTestCase().assertGreaterEqual(1, 0)
        mock_method.assert_called_once()
    
        # CHECK: assert_tensor_greater_equal is invoked appropriately
        with mock.patch.object(ttc.TorchTestCase, "assert_tensor_greater_equal") as mock_method:
            ttc.TorchTestCase().assertGreaterEqual(1, torch.zeros(3))
        mock_method.assert_called_once()

    @staticmethod
    def test_assertLess():
        # CHECK: the original assertLess method is invoked appropriately
        with mock.patch.object(ttc.TorchTestCase, "assertLess") as mock_method:
            ttc.TorchTestCase().assertLess(0, 1)
        mock_method.assert_called_once()
    
        # CHECK: assert_tensor_greater is invoked appropriately
        with mock.patch.object(ttc.TorchTestCase, "assert_tensor_less") as mock_method:
            ttc.TorchTestCase().assertLess(0, torch.ones(3))
        mock_method.assert_called_once()

    @staticmethod
    def test_assertLessEqual():
        # CHECK: the original assertLessEqual method is invoked appropriately
        with mock.patch.object(ttc.TorchTestCase, "assertLessEqual") as mock_method:
            ttc.TorchTestCase().assertLessEqual(0, 1)
        mock_method.assert_called_once()
    
        # CHECK: assert_tensor_less_equal is invoked appropriately
        with mock.patch.object(ttc.TorchTestCase, "assert_tensor_less_equal") as mock_method:
            ttc.TorchTestCase().assertLessEqual(0, torch.ones(3))
        mock_method.assert_called_once()
    
    def test_eps(self):
        # CHECK: the initial value of eps is 0
        self.assertEqual(0, self.test_case.eps)
        
        # CHECK: legal values of eps are stored correctly
        self.test_case.eps = 1e-9
        self.assertEqual(1e-9, self.test_case.eps)
        self.test_case.eps = 1
        self.assertEqual(1, self.test_case.eps)
        self.test_case.eps = 0
        self.assertEqual(0, self.test_case.eps)
        
        # CHECK: providing anything but a real number causes a TypeError
        with self.assertRaises(TypeError):
            self.test_case.eps = "0.001"
        with self.assertRaises(TypeError):
            self.test_case.eps = [0.001]
        
        # CHECK: providing negative numbers causes a ValueError
        with self.assertRaises(ValueError):
            self.test_case.eps = -0.001
        with self.assertRaises(ValueError):
            self.test_case.eps = -1
    
    def test_prepare_tensor_order_comparison(self):
        # CHECK: providing illegally typed args causes a TypeError
        with self.assertRaises(TypeError):
            self.test_case._prepare_tensor_order_comparison(torch.ones(3), "no-tensor")
        with self.assertRaises(TypeError):
            self.test_case._prepare_tensor_order_comparison("no-tensor", torch.ones(3))
        with self.assertRaises(TypeError):
            self.test_case._prepare_tensor_order_comparison(torch.ones(3), nn.Parameter(torch.zeros(3)))
        with self.assertRaises(TypeError):
            self.test_case._prepare_tensor_order_comparison(nn.Parameter(torch.zeros(3)), torch.ones(3))
        
        # CHECK: providing two numbers causes a TypeError
        with self.assertRaises(TypeError):
            self.test_case._prepare_tensor_order_comparison(1, 2)
        
        # CHECK: providing two tensors of different shapes causes a ValueError
        with self.assertRaises(ValueError):
            self.test_case._prepare_tensor_order_comparison(torch.zeros(2), torch.ones(3))
        
        # CHECK: numbers are expanded correctly
        first, second = self.test_case._prepare_tensor_order_comparison(torch.zeros(3), 1)
        self.assertTrue(torch.equal(torch.zeros(3), first))
        self.assertTrue(torch.equal(torch.ones(3), second))
        first, second = self.test_case._prepare_tensor_order_comparison(0, torch.ones(3, 2))
        self.assertTrue(torch.equal(torch.zeros(3, 2), first))
        self.assertTrue(torch.equal(torch.ones(3, 2), second))
        
        # CHECK: no errors occur if two equally shaped tensors are provided
        self.test_case._prepare_tensor_order_comparison(torch.zeros(3, 2, 1), torch.ones(3, 2, 1))

    # noinspection PyArgumentList
    def test_tensor_comparison(self):
        # create several test tensors
        tensor_0 = torch.zeros(3, 3)
        tensor_1 = torch.FloatTensor(
                [
                        [0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]
                ]
        )
        tensor_2 = torch.LongTensor(
                [
                        [1, 1, 1],
                        [4, 4, 4],
                        [7, 7, 7]
                ]
        )
        tensor_3 = torch.ShortTensor([1, 3, 5])
        tensor_4 = torch.IntTensor([5, 2, 5])
        
        # CHECK: the method returns None if the comparison is True
        self.assertIsNone(self.test_case._tensor_comparison(tensor_0, tensor_1, operator.le))
        
        # CHECK: the method provides a list of differing coordinates if the comparison is False
        self.assertEqual(
                [
                        (0, 1),
                        (0, 2),
                        (1, 1),
                        (1, 2),
                        (2, 1),
                        (2, 2)
                ],
                self.test_case._tensor_comparison(tensor_1, tensor_2, operator.lt)
        )
        self.assertEqual(
                [(0,)],
                self.test_case._tensor_comparison(tensor_3, tensor_4, operator.ge)
        )


if __name__ == "__main__":
    unittest.main()
