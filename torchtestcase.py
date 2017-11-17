# -*- coding: utf-8 -*-


import typing
import unittest

import torch

from torch import autograd as ag
from torch import nn
from torch.nn.utils import rnn


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


class TorchTestCase(unittest.TestCase):
    """This class extends ``unittest.TestCase`` with support for equality assertions for various PyTorch classes.
    
    THe ``TorchTestCase`` implements equality assertions for the following types:
    * all types of PyTorch tensors,
    * ``torch.autograd.Variable``,
    * ``torch.nn.Parameter``, and
    * ``torch.nn.utils.rnn.PackedSequence``.
    """

    TENSOR_TYPES = [
            torch.ShortTensor,
            torch.cuda.ShortTensor,
            torch.IntTensor,
            torch.cuda.IntTensor,
            torch.LongTensor,
            torch.cuda.LongTensor,
            torch.HalfTensor,
            torch.cuda.HalfTensor,
            torch.FloatTensor,
            torch.cuda.FloatTensor,
            torch.DoubleTensor,
            torch.cuda.DoubleTensor,
            torch.ByteTensor,
            torch.cuda.ByteTensor,
            torch.CharTensor,
            torch.cuda.CharTensor
    ]
    """list[type]: A list of all different types of PyTorch tensors."""
    
    #  CONSTRUCTOR  ####################################################################################################

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # add equality functions for tensors
        for t in self.TENSOR_TYPES:
            self.addTypeEqualityFunc(t, self.assert_tensor_equal)
        
        # add equality function for variables
        self.addTypeEqualityFunc(ag.Variable, self.assert_variable_equal)

        # add equality function for parameters
        self.addTypeEqualityFunc(nn.Parameter, self.assert_parameter_equal)
        
        # add equality function for packed sequences
        self.addTypeEqualityFunc(torch.nn.utils.rnn.PackedSequence, self.assert_packed_sequence_equal)
    
    #  METHODS  ########################################################################################################
    
    def _fail_with_message(self, msg: typing.Union[str, None], standard_msg: str) -> None:
        """A convenience method that first formats the message to display with ``_formatMessage``, and then invokes
        ``fail``.
        
        Args:
            msg (str or None): The explicit user-defined message.
            standard_msg (str): The standard message created by some assertion method.
        """
        self.fail(self._formatMessage(msg, standard_msg))
    
    def assert_packed_sequence_equal(self, first, second, msg: str=None) -> None:
        """An equality assertion for ``torch.nn.utils.rnn.PackedSequence``s.

        Args:
            first: The first sequence to compare.
            second: The second sequence to compare.
            msg: An optional error message.
        """
        # check whether both args are sequences that contain the same data type
        if not isinstance(first, rnn.PackedSequence):
            self._fail_with_message(msg, "The first argument is not a PackedSequence!")
        if not isinstance(second, rnn.PackedSequence):
            self._fail_with_message(msg, "The second argument is not a PackedSequence!")
        if not isinstance(first.data, type(second.data)):
            self._fail_with_message(
                    msg,
                    "The sequences contain data of different types: a {} is not a {}!".format(
                            type(first.data).__name__,
                            type(second.data).__name__
                    )
            )

        # check whether the sequences' batch sizes and data tensors are equal
        if first.batch_sizes != second.batch_sizes:
            self._fail_with_message(msg, "The sequences have different batch size!")
        if not torch.equal(first.data, second.data):
            self._fail_with_message(msg, "The sequences contain different data!")
    
    def assert_parameter_equal(self, first, second, msg: str=None) -> None:
        """An equality assertion for PyTorch parameters.

        Args:
            first: The first parameter to compare.
            second: The second parameter to compare.
            msg: An optional error message.
        """
        # check whether both args are parameters that contain the same data type
        if not isinstance(first, nn.Parameter):
            self._fail_with_message(msg, "The first argument is not a parameter!")
        if not isinstance(second, nn.Parameter):
            self._fail_with_message(msg, "The second argument is not a parameter!")
        if not isinstance(first.data, type(second.data)):
            self._fail_with_message(
                    msg,
                    "The parameters contain data of different types: a {} is not a {}!".format(
                            type(first.data).__name__,
                            type(second.data).__name__
                    )
            )

        # check whether the parameters' data tensors are equal
        if not torch.equal(first.data, second.data):
            self._fail_with_message(msg, "The parameters contain different data!")
    
    def assert_tensor_equal(self, first, second, msg: str=None) -> None:
        """An equality assertion for PyTorch tensors.
        
        Args:
            first: The first tensor to compare.
            second: The second tensor to compare.
            msg: An optional error message.
        """
        # check whether both args are tensors of the same type
        if type(first) not in self.TENSOR_TYPES:
            self._fail_with_message(msg, "The first argument is not a tensor!")
        if type(second) not in self.TENSOR_TYPES:
            self._fail_with_message(msg, "The second argument is not a tensor!")
        if not isinstance(first, type(second)):
            self._fail_with_message(msg, "A {} is not a {}!".format(type(first).__name__, type(second).__name__))
        
        # check whether tensors are equal
        if not torch.equal(first, second):
            self._fail_with_message(msg, "The tensors are different!")

    def assert_variable_equal(self, first, second, msg: str=None) -> None:
        """An equality assertion for PyTorch variables.

        Args:
            first: The first variable to compare.
            second: The second variable to compare.
            msg: An optional error message.
        """
        # check whether both args are variables that contain the same data type
        if not isinstance(first, ag.Variable):
            self._fail_with_message(msg, "The first argument is not a variable!")
        if not isinstance(second, ag.Variable):
            self._fail_with_message(msg, "The second argument is not a variable!")
        if not isinstance(first.data, type(second.data)):
            self._fail_with_message(
                    msg,
                    "The variables contain data of different types: a {} is not a {}!".format(
                            type(first.data).__name__,
                            type(second.data).__name__
                    )
            )

        # check whether the variables' data tensors are equal
        if not torch.equal(first.data, second.data):
            self._fail_with_message(msg, "The variables contain different data!")
