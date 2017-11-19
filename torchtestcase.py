# -*- coding: utf-8 -*-


import numbers
import operator
import typing
import unittest

import numpy as np
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
    """This class extends ``unittest.TestCase`` such that some of the available assertions support instances of various
    PyTorch classes.
    
    ``TorchTestCase`` provides the following PyTorch-specific functionality:
    * ``assertEqual`` supports all kinds of PyTorch tensors as well as instances of ``torch.autograd.Variable``,
      ``torch.nn.Parameter``, and ``torch.nn.utils.rnn.PackedSequence``.
    * ``assertGreater``, ``assertGreaterEqual``, ``assertLess``, and ``assertLessEqual`` support all kinds of PyTorch
      tensors except ``CharTensor``s as well as instances of ``torch.autograd.Variable`` and ``torch.nn.Parameter``.
      Furthermore, these assertions allow for comparing tensors to numbers. Notice, however, that neither of the
      mentioned assertions performs any kind of type check in the sense that it is possible to compare a
      ``FloatTensor`` with a ``Parameter``, for example.
    """
    
    ORDER_ASSERTION_TYPES = [
            torch.ByteTensor,
            torch.cuda.ByteTensor,
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
            torch.cuda.DoubleTensor
    ]
    """list[type]: A list of all types of PyTorch tensors that are supported by order assertions, like lower-than."""

    TENSOR_TYPES = [
            torch.ByteTensor,
            torch.cuda.ByteTensor,
            torch.CharTensor,
            torch.cuda.CharTensor,
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
            torch.cuda.DoubleTensor
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
    
    @classmethod
    def _prepare_tensor_order_comparison(cls, first, second) -> typing.Tuple[typing.Any, typing.Any]:
        """This method prepares tensors for subsequent order comparisons.
        
        The preparation includes the following steps:
        1. check that both args are either a tensor or a number,
        2. check that at least one of them is a tensor,
        3. if both args are tensors, then check whether they have the same shape, and
        4. turn any provided number into an according tensor of appropriate shape.
        
        Notice that order comparison support all kinds of PyTorch tensors except ``CharTensor``s, which is why this
        method raises a ``TypeError`` if a ``CharTensor`` is provided.
        
        Args:
            first: The first tensor or number to prepare.
            second: The second tensor or number to prepare.
        
        Returns:
             tuple: The method simply returns the prepared args in the same order that they were provided.
        
        Raises:
            TypeError: If any of ``first`` or ``second`` is neither a supported kind of tensor nor a number, or if
                both args are numbers.
            ValueErrors: If ``first`` and ``second`` are tensors of different shape.
        """
        # ensure that both args are either a tensor or a number
        if type(first) not in cls.ORDER_ASSERTION_TYPES and not isinstance(first, numbers.Real):
            raise TypeError("The first argument is neither a supported type fo tensor nor a number!")
        if type(second) not in cls.ORDER_ASSERTION_TYPES and not isinstance(second, numbers.Real):
            raise TypeError("The second argument is neither a supported type of tensor nor a number!")
        
        # if both args are tensor then check whether they have the same shape
        if (
                type(first) in cls.ORDER_ASSERTION_TYPES and
                type(second) in cls.ORDER_ASSERTION_TYPES and
                first.shape != second.shape
        ):
            raise ValueError("The arguments must not be tensors of different shapes!")
        
        # turn first argument into tensor if it is a number
        if isinstance(first, numbers.Real):
            first = float(first)
            if isinstance(second, numbers.Real):
                raise TypeError("At least one the arguments has to be a tensor!")
            else:
                first = torch.ones(second.shape) * first

        # turn second argument into tensor if it is a number
        if isinstance(second, numbers.Real):
            second = float(second)
            if isinstance(first, numbers.Real):
                raise TypeError("At least one the arguments has to be a tensor!")
            else:
                second = torch.ones(first.shape) * second
        
        return first, second
    
    def _tensor_aware_assertion(
            self,
            tensor_assertion: typing.Callable,
            default_assertion: typing.Callable,
            first,
            second,
            msg: typing.Optional[str]
    ) -> None:
        """Invokes either a tensor-specific version of an assertion or the original implementation provided by
        ``unittest.TestCase``.
        
        This method assumes that a function that implements some assertion has to be invoked as
        
            some-assertion(first, second, msg=msg)
        
        If either ``first`` or ``second`` is a PyTorch Tensor, then we invoke ``tensor_assertion``, and otherwise
        we use ``default_assertion``.
        
        Args:
            tensor_assertion (callable): The tensor-specific implementation of an assertion.
            default_assertion (callable): The default implementation of the same assertion.
            first: The first arg to pass to the assertion method.
            second: The second arg to pass to the assertion method.
            msg (str): Passed to the assertion method as keyword arg ``msg``.
        """
        # check whether any of the args is a tensor/variable/parameter
        # if yes -> call tensor-specific assertion check
        all_tensor_types = self.TENSOR_TYPES + [ag.Variable, nn.Parameter]
        if type(first) in all_tensor_types or type(second) in all_tensor_types:
        
            # turn variables/parameters into tensors
            if isinstance(first, ag.Variable) or isinstance(first, nn.Parameter):
                first = first.data
            if isinstance(second, ag.Variable) or isinstance(second, nn.Parameter):
                second = second.data
        
            # invoke assertion check for tensors
            tensor_assertion(first, second, msg=msg)
    
        # call original method for checking the assertion
        else:
            default_assertion(first, second, msg=msg)
    
    @staticmethod
    def _tensor_comparison(
            first,
            second,
            comp_op: typing.Callable
    ) -> typing.Optional[typing.List[typing.Tuple[int, ...]]]:
        """Compares two PyTorch tensors element-wisely by means of the provided comparison operator.
        
        The provided tensors may be of any, possibly different types of PyTorch tensors except ``CharTensor``. They do
        have to be of equal shape, though. Notice further that this method expects actual tensors as opposed to PyTorch
        ``Variable``s or ``Parameter``s.
        
        Args:
            first: The first PyTorch tensor to compare.
            second: The second PyTorch tensor to compare.
            comp_op: The comparison operator to use.
        
        Returns:
            ``None``, if the comparison evaluates to ``True`` for all coordinates, and a list of positions, i.e., tuples
             of ``int`` values, where it does not, otherwise.
        """
        # turn both tensors into numpy arrays
        first = first.cpu().numpy()
        second = second.cpu().numpy()
        
        # compare both args
        comp = comp_op(first, second)
        
        # if comparison yields true for each entry -> nothing else to do
        if comp.all():
            return None

        # retrieve all coordinates where the comparison evaluated to False
        index_lists = [list(l) for l in np.where(np.invert(comp))]
        coord_list = list(zip(*index_lists))
        
        return coord_list
    
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
    
    def assert_tensor_greater(self, first, second, msg: str=None) -> None:
        """A greater-than assertion for PyTorch tensors.
        
        Notice that ``CharTensor``s are not supported by this method, and one of the args may be a number rather than a
        tensor.
        
        Args:
            first: The first tensor (or number) to compare.
            second: The second tensor (or number) to compare.
            msg: An optional error message.
        """
        try:
            first, second = self._prepare_tensor_order_comparison(first, second)
        except (TypeError, ValueError) as e:
            self._fail_with_message(msg, str(e))
        
        fails = self._tensor_comparison(first, second, operator.gt)
        if fails is not None:
            std_msg = (
                    "The first tensor is not greater than the second!\n"
                    "\n"
                    "The assertion fails at the following positions:"
            )
            for pos in fails:
                std_msg += "\n- ({}): {} <= {}".format(
                        ", ".join([str(x) for x in pos]),
                        first[pos],
                        second[pos]
                )
            self._fail_with_message(msg, std_msg)

    def assert_tensor_greater_equal(self, first, second, msg: str=None) -> None:
        """A greater-than-or-equal-to assertion for PyTorch tensors.

        Notice that ``CharTensor``s are not supported by this method, and one of the args may be a number rather than a
        tensor.

        Args:
            first: The first tensor (or number) to compare.
            second: The second tensor (or number) to compare.
            msg: An optional error message.
        """
        try:
            first, second = self._prepare_tensor_order_comparison(first, second)
        except (TypeError, ValueError) as e:
            self._fail_with_message(msg, str(e))
    
        fails = self._tensor_comparison(first, second, operator.ge)
        if fails is not None:
            std_msg = (
                    "The first tensor is not greater than or equal to the second!\n"
                    "\n"
                    "The assertion fails at the following positions:"
            )
            for pos in fails:
                std_msg += "\n- ({}): {} < {}".format(
                        ", ".join([str(x) for x in pos]),
                        first[pos],
                        second[pos]
                )
            self._fail_with_message(msg, std_msg)

    def assert_tensor_less(self, first, second, msg: str = None) -> None:
        """A less-than assertion for PyTorch tensors.

        Notice that ``CharTensor``s are not supported by this method, and one of the args may be a number rather than a
        tensor.

        Args:
            first: The first tensor (or number) to compare.
            second: The second tensor (or number) to compare.
            msg: An optional error message.
        """
        try:
            first, second = self._prepare_tensor_order_comparison(first, second)
        except (TypeError, ValueError) as e:
            self._fail_with_message(msg, str(e))
    
        fails = self._tensor_comparison(first, second, operator.lt)
        if fails is not None:
            std_msg = (
                    "The first tensor is not less than the second!\n"
                    "\n"
                    "The assertion fails at the following positions:"
            )
            for pos in fails:
                std_msg += "\n- ({}): {} > {}".format(
                        ", ".join([str(x) for x in pos]),
                        first[pos],
                        second[pos]
                )
            self._fail_with_message(msg, std_msg)

    def assert_tensor_less_equal(self, first, second, msg: str = None) -> None:
        """A less-than-or-equal-to assertion for PyTorch tensors.

        Notice that ``CharTensor``s are not supported by this method, and one of the args may be a number rather than a
        tensor.

        Args:
            first: The first tensor (or number) to compare.
            second: The second tensor (or number) to compare.
            msg: An optional error message.
        """
        try:
            first, second = self._prepare_tensor_order_comparison(first, second)
        except (TypeError, ValueError) as e:
            self._fail_with_message(msg, str(e))
    
        fails = self._tensor_comparison(first, second, operator.le)
        if fails is not None:
            std_msg = (
                    "The first tensor is not less than or equal to the second!\n"
                    "\n"
                    "The assertion fails at the following positions:"
            )
            for pos in fails:
                std_msg += "\n- ({}): {} > {}".format(
                        ", ".join([str(x) for x in pos]),
                        first[pos],
                        second[pos]
                )
            self._fail_with_message(msg, std_msg)
    
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
    
    def assertGreater(self, a, b, msg=None):
        self._tensor_aware_assertion(self.assert_tensor_greater, super().assertGreater, a, b, msg)

    def assertGreaterEqual(self, a, b, msg=None):
        self._tensor_aware_assertion(self.assert_tensor_greater_equal, super().assertGreaterEqual, a, b, msg)
    
    def assertLess(self, a, b, msg=None):
        self._tensor_aware_assertion(self.assert_tensor_less, super().assertLess, a, b, msg)

    def assertLessEqual(self, a, b, msg=None):
        self._tensor_aware_assertion(self.assert_tensor_less_equal, super().assertLessEqual, a, b, msg)
