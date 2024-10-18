import pytest
from toonygrad import Tensor

def test_tensor_addition():
    out = Tensor([1., 2, 3]) + Tensor([4., 5, 6])
    out.tolist == [5.0, 7.0, 9.0]

def test_tensor_def_output():
    Tensor([]) is not None
    print(Tensor([]))


def test_tensor_addition_tensor_none():
    with pytest.raises(ValueError):
      Tensor([1., 2, 3]) + Tensor([])
  