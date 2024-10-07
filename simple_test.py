#!/usr/bin/env python3
from toonygrad import Tensor

if __name__ == "__main__":
  from toonygrad.nn.state import get_parameters
  from beautiful_mnist import Model

  model = Model()
  for p in get_parameters(model): p.replace(Tensor.empty(p.shape))
  model(Tensor.empty(1, 1, 28, 28)).realize()
  exit(0)

  a = Tensor([2])
  b = Tensor([3])
  print((a+b).tolist())
