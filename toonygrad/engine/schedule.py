from typing import List
from toonygrad.ops import UOp, graph_rewrite, PatternMatcher, UPat, UOps

class ScheduleItem:
  pass

pm = PatternMatcher([
  (UPat(UOps.SWIZZLE, src=(UPat(UOps.SWIZZLE, name="s0"),), name="s1"),
   lambda s0,s1: UOp(UOps.SWIZZLE, s1.dtype, s0.src, s0.arg+s1.arg)),
])

def create_schedule_with_vars(sched:List[UOp]):
  # TODO: should the input be a SINK?
  sink = UOp.sink(*sched)
  sink = graph_rewrite(sink, pm)
  #print(sink)
  return [], {}
