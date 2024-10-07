from typing import cast, Tuple
from toonygrad.ops import UOp, UOps
from toonygrad.shape.symbolic import sint
from toonygrad.shape.shapetracker import ShapeTracker
from toonygrad.helpers import all_same, unwrap, prod

class LazyBuffer(UOp):
  @staticmethod
  def metaop(op, shape, dtype, device):
    print(op, shape, dtype, device)
    return LazyBuffer(UOps.BUFFER, dtype, (ShapeTracker.from_shape(shape).to_uop(),))
