#!/usr/bin/env python3
from toonygrad import Tensor

if __name__ == "__main__":
  a = Tensor([2])
  b = Tensor([3])
  print((a+b).tolist())
