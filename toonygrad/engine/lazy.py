from __future__ import annotations
from typing import cast, Tuple, Dict, Optional
from toonygrad.ops import UOp, UOps, MetaOps, REDUCE_ALU, MathTrait
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
  def shape(self):
    assert self.uop.shape is not None, self.uop
    return self.uop.shape
  @property
  def device(self):
    assert self.uop.device is not None, self.uop
    return self.uop.device

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

  # support for very basic broadcasting
  def alu(self, arg, *src):
    srcs = (self,)+src
    shapes = [x.uop.shape for x in srcs]
    non_none_shapes = [x for x in shapes if x is not None]
    assert len(non_none_shapes) != 0 and all_same(non_none_shapes)
    my_shape = non_none_shapes[0]
    return LazyBuffer(UOp(UOps.ALU, self.dtype, tuple(x.uop if x.uop.shape == my_shape else x.reshape((1,)*len(my_shape)).expand(my_shape).uop for x in srcs), arg))

  # simple proxy functions
  def copy_to_device(self, device): return LazyBuffer(UOp(UOps.COPY, self.dtype, (self.uop,), device))
  def r(self, op, axis): return LazyBuffer(UOp(UOps.REDUCE_AXIS, self.dtype, (self.uop,), (REDUCE_ALU[op], axis)))
  def cast(self, dtype): return LazyBuffer(self.uop.cast(dtype))
  def bitcast(self, dtype): return LazyBuffer(self.uop.bitcast(dtype))
  def contiguous(self): return LazyBuffer(UOp(UOps.CONTIGUOUS, self.dtype, (self.uop,)))
  def const_like(self, b): return LazyBuffer(self.uop.const_like(b))

  @property
  def lbs(self): return [self]

  # movement functions
  def _swizzle(self, method, arg):
    st = ShapeTracker.from_shape(tuple() if self.uop.shape is None else self.uop.shape)
    return LazyBuffer(UOp(UOps.SWIZZLE, self.dtype, (self.uop,), st.__getattribute__(method)(arg)))
  def reshape(self, shape): return self._swizzle('reshape', shape)
  def expand(self, shape): return self._swizzle('expand', shape)
  def permute(self, arg): return self._swizzle('permute', arg)
  def pad(self, arg): return self._swizzle('pad', arg)
  def shrink(self, arg): return self._swizzle('shrink', arg)
  def stride(self, arg): return self._swizzle('stride', arg)
