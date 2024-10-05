from __future__ import annotations
from typing import Tuple
from toonygrad.ops import UOp, MetaOps
from toonygrad.shape.symbolic import sint
from toonygrad.dtype import DTypeLike, to_dtype

class LazyBuffer(UOp):
  @staticmethod
  def metaop(op, shape:Tuple[sint,...], dtype:DTypeLike, device:str, arg=None, src:Tuple[LazyBuffer, ...]=(), enable_cache=False) -> LazyBuffer:
    if op is MetaOps.EMPTY: raise RuntimeError("why do we have this?")

    pass
