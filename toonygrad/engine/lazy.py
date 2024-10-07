from __future__ import annotations
from typing import cast, Tuple, Dict
from toonygrad.ops import UOp, UOps, MetaOps, buffers
from toonygrad.shape.symbolic import sint
from toonygrad.shape.shapetracker import ShapeTracker
from toonygrad.helpers import all_same, unwrap, prod

class LazyBuffer(UOp):
  buffer_num = -1

  @staticmethod
  def get_buffer(x:UOp):
    assert x in buffers, "need to realize to get buffer"
    return buffers[x]

  @staticmethod
  def metaop(op, shape, dtype, device, arg=None, src=None):
    if op is MetaOps.CONST:
      return UOp.const(dtype, arg).reshape(shape)
    if op is MetaOps.EMPTY:
      LazyBuffer.buffer_num += 1
      return LazyBuffer(UOps.BUFFER, dtype, arg=(device, prod(shape), LazyBuffer.buffer_num)).reshape(shape)
    raise Exception(f"unhandled MetaOp {op}")
