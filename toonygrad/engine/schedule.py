from typing import List
from toonygrad.ops import UOp, graph_rewrite, PatternMatcher, UPat, UOps, symbolic

class ScheduleItem:
  pass

pm = symbolic+PatternMatcher([
  # merge swizzle
  (UPat(UOps.SWIZZLE, src=(UPat(UOps.SWIZZLE, name="s0"),), name="s1"),
   lambda s0,s1: UOp(UOps.SWIZZLE, s1.dtype, s0.src, s0.arg+s1.arg)),
  # swizzle before ALU
  (UPat(UOps.SWIZZLE, src=(UPat(UOps.ALU, name="alu"),), name="s"),
    lambda alu,s: UOp(UOps.ALU, alu.dtype,
                      tuple(UOp(UOps.SWIZZLE, x.dtype, (x,), s.arg) for x in alu.src), alu.arg)),
  # const + maskless swizzle = const
  (UPat(UOps.SWIZZLE, src=(UPat.cvar('c'),), name="s"),
    lambda s,c: c if all(x.mask is None for x in s.st.views) else None),
])

def create_schedule_with_vars(sched:List[UOp]):
  # TODO: should the input be a SINK?
  sink = UOp.sink(*sched)
  sink = graph_rewrite(sink, pm)
  #print(sink)
  return [], {}
