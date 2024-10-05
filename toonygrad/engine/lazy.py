from __future__ import annotations
from typing import Tuple
from tinygrad.ops import UOp, MetaOps
from tinygrad.shape.symbolic import sint
from tinygrad.dtype import DTypeLike, to_dtype

class LazyBuffer(UOp):
  @staticmethod
  def metaop(op, shape:Tuple[sint,...], dtype:DTypeLike, device:str, arg=None, src:Tuple[LazyBuffer, ...]=(), enable_cache=False) -> LazyBuffer:
    print("metaop", op)
    pass
