from __future__ import annotations
from typing import cast, Tuple, Dict, Optional
from toonygrad.ops import UOp, UOps, MetaOps, buffers, REDUCE_ALU, MathTrait
from toonygrad.shape.symbolic import Variable, sint
from toonygrad.shape.shapetracker import ShapeTracker
from toonygrad.helpers import all_same, unwrap, prod
from toonygrad.device import Buffer

# LazyBuffer is where UOp are bound to Buffer
class LazyBuffer(MathTrait):
  buffer_num = -1
  def __init__(self, uop):
    self.uop: UOp = uop
    self._buffer: Optional[Buffer] = None

  # TODO: this is broken
  def is_realized(self): return self._buffer is not None

  @property
  def dtype(self): return self.uop.dtype
  @property
  def size(self): return self.uop.size
  @property
  def shape(self): return self.uop.shape
  @property
  def device(self): return self.uop.device

  @property
  def buffer(self) -> Buffer:
    if self._buffer is not None: return self._buffer
    uop = self.uop if self.uop.op is UOps.BUFFER else self.uop.src[0]
    assert uop.op == UOps.BUFFER
    self._buffer = Buffer(uop.arg[0], uop.arg[1], self.dtype)
    return self._buffer

  @staticmethod
  def get_buffer(x:UOp):
    assert x in buffers, "need to realize to get buffer"
    return buffers[x]

  @staticmethod
  def metaop(op, shape, dtype, device, arg=None, src=None):
    if op is MetaOps.CONST:
      return LazyBuffer(UOp.const(dtype, arg)).copy_to_device(device).reshape(shape)
    if op is MetaOps.EMPTY:
      LazyBuffer.buffer_num += 1
      return LazyBuffer(UOp(UOps.BUFFER, dtype, arg=(device, prod(shape), LazyBuffer.buffer_num))).reshape(shape)
    raise Exception(f"unhandled MetaOp {op}")

  # proxy functions
  def copy_to_device(self, device): return LazyBuffer(UOp(UOps.COPY, self.dtype, (self.uop,), device))
  def r(self, op, axis): return LazyBuffer(UOp(UOps.REDUCE_AXIS, self.dtype, (self.uop,), (REDUCE_ALU[op], axis)))
  def alu(self, arg, *src): return LazyBuffer(UOp(UOps.ALU, self.dtype, tuple(x.uop for x in src), arg))
  def cast(self, dtype): return LazyBuffer(self.uop.cast(dtype))
  def bitcast(self, dtype): return LazyBuffer(self.uop.bitcast(dtype))
  def contiguous(self): return LazyBuffer(UOp(UOps.CONTIGUOUS, self.dtype, (self.uop,)))
  def const_like(self, b): return LazyBuffer(self.uop.const_like(b))

  # movement functions
  def _swizzle(self, method, arg):
    return LazyBuffer(UOp(UOps.SWIZZLE, self.dtype, (self.uop,), self.uop.st_shape.__getattribute__(method)(arg)))
  def reshape(self, shape): return self._swizzle('reshape', shape)
  def expand(self, shape): return self._swizzle('expand', shape)
  def permute(self, arg): return self._swizzle('permute', arg)
  def pad(self, arg): return self._swizzle('pad', arg)
  def shrink(self, arg): return self._swizzle('shrink', arg)
  def stride(self, arg): return self._swizzle('stride', arg)
