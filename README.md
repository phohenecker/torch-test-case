torch-test-case
===============


Using Python's [`unittest`](https://docs.python.org/3/library/unittest.html) package turns out to be cumbersome when we
are working with [PyTorch](http://pytorch.org/) and need to write assertions that include tensors, parameters, and so
forth.
The main reason for this is that PyTorch tensors are compared element-wise by default, which is why assertions provided
by the class [`unittest.TestCase`](https://docs.python.org/3/library/unittest.html#unittest.TestCase) do not work
out-of-the-box.
A possible workaround is to use
[`TestCase.assertTrue`](https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertTrue) for any assertion
that we need to make, yet this commonly leads to convoluted code that is hard to read and maintain.

The module `torchtestcase` defines the class `TorchTestCase`, which extends `unittest.TestCase` such that many
assertions support instances of various PyTorch classes.

**Update**:
Version 2018.1 has been released, and supports PyTorch 0.4 now.


Installation
------------

This module can be installed from PyPI:
```
pip install torchtestcase
```


PyTorch Assertions
------------------

This section describes those assertions provided by the class `TorchTestCase` that support PyTorch.
If you are not familiar with the package `unittest`, then read about it first
[here](https://docs.python.org/3/library/unittest.html).

**Notice**:
With the release of PyTorch 0.4.0, tensors and variables have been merged, which means that `Variable`s are treated just
like any other tensors, and thus there is no need to make use of the class `torch.autograd.Variable` anymore.
Accordingly, assertions for `Variable`s in particular have been removed in version 2018.1 of `torchtestcase`.


### 1. Equality Assertions

(`assertEqual`, `assertNotEqual`)

Equality assertions support objects that are any kind of PyTorch tensors as well as instances of `torch.nn.Parameter`
and `torch.nn.utils.rnn.PackedSequence`.
Notice, however, that an `AssertionError` is raised if the compared objects are instances of different types:
```python
self.assertEqual(torch.zeros(4), nn.Parameter(torch.zeros(4))  # -> AssertionError
```


### 2. Order Assertions

(`assertGreater`, `assertGreaterEqual`, `assertLess`, `assertLessEqual`)

In general, order assertions are assumed to be fulfilled if they hold element-wise.
For example:
```pyhton
x = torch.FloatTensor([0, 0, 1])
y = torch.FloatTensor([1, 1, 1])
self.assertLessEqual(x, y)  # -> no AssertionError
self.assertLess(x, y)       # -> AssertionError
```
In addition, it is possible to compare tensors or `Parameters` to a number, in which case each element of the considered
data tensor is compared to the same.
For example, if we want to ensure that every element of a tensor lies in the unit interval, then we may use the
following assertions:
```python
self.assertGreaterEqual(some_tensor, 0)
self.assertLessEqual(some_tensor, 1)
```
When we make order assertions, then we usually do not care about the actual types of the objects involved.
Therefore, it is possible to compare different kinds of tensors with each other as well as with `Parameter`s:
```python
self.assertLess(torch.zeros(3), nn.Parameter(torch.ones(3)))  # -> no AssertionError
```
