  # *** this was LazyBuffer ***

  def copy_to_device(self, device): return UOp(UOps.COPY, self.dtype, (self,), device)
  def r(self, op, axis): return UOp(UOps.REDUCE_AXIS, self.dtype, (self,), (REDUCE_ALU[op], axis, tuple(self.shape[x] for x in axis)))
  def contiguous(self): return UOp(UOps.CONTIGUOUS, self.dtype, (self,))

  @property
  def lbs(self): return [self]

  @functools.cached_property
  def device(self):
    if self.op is UOps.BUFFER: return self.arg[0]
    if self.op is UOps.COPY: return self.arg
    devices = [x.device for x in self.src]
    non_none_devices = [x for x in devices if x is not None]
    if len(non_none_devices) == 0: return None
    assert all_same(non_none_devices), f"device mismatch {non_none_devices}"
    return non_none_devices[0]

  @functools.cached_property
  def shape(self):
    if self.op is UOps.VIEW:
      st = self.st
      assert st is not None
      return st.shape
    if self.op is UOps.BUFFER: return (self.arg[1],)
    if self.op in {UOps.LOAD, UOps.STORE}: return self.src[1].shape
    if self.op is UOps.CONST: return None
    shapes = [x.shape for x in self.src]
    non_none_shapes = [x for x in shapes if x is not None]
    if len(non_none_shapes) == 0: return None
    assert all_same(non_none_shapes), f"shape mismatch {non_none_shapes}, {self}"
    ret = list(non_none_shapes[0])
    if self.op is UOps.REDUCE_AXIS:
      for axis in self.arg[1]: ret[axis] = 1
    return tuple(ret)

  @property
  def size(self) -> sint:
    assert self.shape is not None, f"no size for {self}"
    return prod(self.shape)

  @property
  def buffer(self) -> Buffer:
    from toonygrad.device import Buffer
    if (ret:=buffers.get(self)) is not None: return ret
    if self.op is UOps.VIEW:
      assert self.st.contiguous == True, "VIEW only works here if it's contiguous"
      return self.src[0].buffer
    assert self.op == UOps.BUFFER, f"no buffer on {self.op}"
    buffers[self] = ret = Buffer(self.arg[0], self.arg[1], self.dtype)
    return ret
  def is_realized(self): return self in buffers

  buffer_num = -1
  @staticmethod
  def new_buffer(dtype, device, size):
    UOp.buffer_num += 1
    return UOp(UOps.BUFFER, dtype, arg=(device, size, UOp.buffer_num))

  @staticmethod
  def metaop(op, shape, dtype, device, arg=None, src=None):
    if op is MetaOps.CONST: return UOp.const(dtype, arg).copy_to_device(device).reshape(shape)
    if op is MetaOps.EMPTY: return UOp.new_buffer(dtype, device, prod(shape)).reshape(shape)
    raise Exception(f"unhandled MetaOp {op}")

  # movement functions
  def _view(self, method, arg):
    if self.op is UOps.VIEW:
      return UOp(UOps.VIEW, self.dtype, (self.src[0],), self.st.__getattribute__(method)(arg))
    else:
      from toonygrad.shape.shapetracker import ShapeTracker
      st = ShapeTracker.from_shape(tuple() if self.shape is None else self.shape)
      return UOp(UOps.VIEW, self.dtype, (self,), st.__getattribute__(method)(arg))
  def reshape(self, shape): return self._view('reshape', shape)
  def expand(self, shape): return self._view('expand', shape)
  def permute(self, arg): return self._view('permute', arg)
  def pad(self, arg): return self._view('pad', arg)
  def shrink(self, arg): return self._view('shrink', arg)
  def stride(self, arg): return self._view('stride', arg)

  # hacks for srcs deleting
  @property
  def srcs(self): return None
  @srcs.deleter
  def srcs(self): pass

if TYPE_CHECKING:
  from toonygrad.device import Buffer
buffers: Dict[UOp, Buffer] = {}
