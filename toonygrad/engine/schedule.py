from typing import List
from toonygrad.ops import UOp, graph_rewrite, PatternMatcher, UPat, UOps, symbolic, track_rewrites
from toonygrad.engine.lazy import LazyBuffer

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

def append_kernel(k:List[UOp], base:UOp, x:UOp):
  k.append(x.sink())
  return base.replace(src=())
break_sched = PatternMatcher([
  (UPat(UOps.SWIZZLE, src=(UPat.var('x'),), name="base"), append_kernel),
])

@track_rewrites
def _schedule_rewrite(sink):
  sink = graph_rewrite(sink, pm)
  sched = []
  sched.append(graph_rewrite(sink, break_sched, []))
  return sched

def create_schedule_with_vars(sched:List[UOp]):
  # TODO: should the input be a SINK?
  sched = _schedule_rewrite(UOp.sink(*sched))
  print(len(sched))
  #print(sink)
  return sched, {}
