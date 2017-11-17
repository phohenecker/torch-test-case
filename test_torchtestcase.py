#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import unittest

import torch

from torch import autograd as ag
from torch import nn
from torch.nn.utils import rnn

import torchtestcase as ttc


__author__ = "Patrick Hohenecker"
__copyright__ = (
        "Copyright (c) 2017 Patrick Hohenecker\n"
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
__version__ = "2017.1"
__date__ = "Nov 17, 2017"
__maintainer__ = "Patrick Hohenecker"
__email__ = "mail@paho.at"
__status__ = "Development"


class TorchTestCaseTest(unittest.TestCase):
    
    def setUp(self):
        self.test_case = ttc.TorchTestCase()

    # noinspection PyArgumentList
    def test_assert_equals(self):
        # CHECK: assert_tensor_equal is invoked appropriately
        self.test_case.assertEqual(
                torch.zeros(3),
                torch.zeros(3)
        )
        with self.assertRaises(AssertionError):
            self.test_case.assertEqual(
                    torch.zeros(3),
                    torch.ones(3)
            )

        # CHECK: assert_variable_equal is invoked appropriately
        self.test_case.assertEqual(
                ag.Variable(torch.zeros(3)),
                ag.Variable(torch.zeros(3))
        )
        with self.assertRaises(AssertionError):
            self.test_case.assertEqual(
                    ag.Variable(torch.zeros(3)),
                    ag.Variable(torch.ones(3))
            )

        # CHECK: assert_parameter_equal is invoked appropriately
        self.test_case.assertEqual(
                nn.Parameter(torch.zeros(3)),
                nn.Parameter(torch.zeros(3))
        )
        with self.assertRaises(AssertionError):
            self.test_case.assertEqual(
                    nn.Parameter(torch.zeros(3)),
                    nn.Parameter(torch.ones(3))
            )
        
        # CHECK: assert_packed_sequence_equal is invoked appropriately
        self.test_case.assertEqual(
                rnn.pack_padded_sequence(torch.FloatTensor([[1, 1], [1, 0]]), [2, 1]),
                rnn.pack_padded_sequence(torch.FloatTensor([[1, 1], [1, 0]]), [2, 1])
        )
        with self.assertRaises(AssertionError):
            self.test_case.assertEqual(
                    rnn.pack_padded_sequence(torch.FloatTensor([[1, 1], [1, 0]]), [2, 1]),
                    rnn.pack_padded_sequence(torch.FloatTensor([[1, 1], [2, 0]]), [2, 1])
            )

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
                    nn.Parameter(torch.LongTensor([0]))
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
        
        # CHECK: no errors are raised for equal tensors
        self.test_case.assert_tensor_equal(torch.zeros(3), torch.zeros(3))

    # noinspection PyArgumentList
    def test_assert_variable_equal(self):
        # CHECK: the assertion fails if either of the args is not a variable
        with self.assertRaises(AssertionError):
            self.test_case.assert_variable_equal("no tensor", ag.Variable(torch.zeros(3)))
        with self.assertRaises(AssertionError):
            self.test_case.assert_variable_equal(ag.Variable(torch.zeros(3)), torch.zeros(3))
    
        # CHECK: the assertion fails if the args contain data of different types
        with self.assertRaises(AssertionError):
            self.test_case.assert_variable_equal(
                    ag.Variable(torch.FloatTensor([0])),
                    ag.Variable(torch.LongTensor([0]))
            )
    
        # CHECK: the assertion fails if the args differ in values or shape
        with self.assertRaises(AssertionError):
            self.test_case.assert_variable_equal(
                    ag.Variable(torch.FloatTensor([0, 1])),
                    ag.Variable(torch.FloatTensor([0, 2]))
            )
        with self.assertRaises(AssertionError):
            self.test_case.assert_variable_equal(
                    ag.Variable(torch.FloatTensor([0, 1])),
                    ag.Variable(torch.FloatTensor([0, 1, 2]))
            )
        with self.assertRaises(AssertionError):
            self.test_case.assert_variable_equal(
                    ag.Variable(torch.FloatTensor([0, 1])),
                    ag.Variable(torch.FloatTensor([[0, 1]]))
            )
    
        # CHECK: no errors are raised for equal variables
        self.test_case.assert_variable_equal(
                ag.Variable(torch.zeros(3)),
                ag.Variable(torch.zeros(3))
        )


if __name__ == "__main__":
    unittest.main()
