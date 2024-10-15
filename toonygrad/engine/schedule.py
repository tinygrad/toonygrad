from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from toonygrad.helpers import Context
from toonygrad.ops import UOp, graph_rewrite, PatternMatcher, UPat, UOps, symbolic, track_rewrites, buffers
from toonygrad.engine.lazy import LazyBuffer
from toonygrad.shape.symbolic import Variable
from toonygrad.shape.shapetracker import ShapeTracker
from toonygrad.device import Buffer

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: Tuple[Buffer, ...]

pm_merge_views_and_consts = PatternMatcher([
  # merge VIEW
  (UPat(UOps.VIEW, src=(UPat(UOps.VIEW, name="s0"),), name="s1"),
   lambda s0,s1: UOp(UOps.VIEW, s1.dtype, s0.src, s0.arg+s1.arg)),
  # const + copy = const
  (UPat(UOps.COPY, src=(UPat.cvar('c'),)), lambda c: c),
  # const + maskless swizzle = const
  (UPat(UOps.VIEW, src=(UPat.cvar('c'),), name="s"),
    lambda s,c: c if all(x.mask is None for x in s.st.views) else None),
])

pm_push_views = symbolic+pm_merge_views_and_consts+PatternMatcher([
  # VIEW before ALU
  (UPat(UOps.VIEW, src=(UPat(UOps.ALU, name="alu"),), name="s"),
    lambda alu,s: UOp(UOps.ALU, alu.dtype,
                      tuple(UOp(UOps.VIEW, x.dtype, (x,), s.arg) for x in alu.src), alu.arg)),
  # don't need CONTIGUOUS any more
  (UPat(UOps.CONTIGUOUS, src=(UPat.var('x'),)), lambda x: x),
])

# *********

def append_buffer(bufs:List[Buffer], buf:UOp, view:Optional[UOp]=None, to_store:Optional[UOp]=None):
  if buf.buffer not in bufs: bufs.append(buf.buffer)
  dg = UOp(UOps.DEFINE_GLOBAL, buf.dtype.ptr(), (), bufs.index(buf.buffer))
  if view is not None: return UOp.load(dg, view.replace(src=()), dtype=buf.dtype)
  if to_store is not None: return UOp.store(dg, ShapeTracker.from_shape(to_store.shape).to_uop(), to_store)

enumerate_bufs = PatternMatcher([
  (UPat(UOps.VIEW, src=(UPat(UOps.BUFFER, src=(), name="buf"),), name="view"), append_buffer),
  (UPat(UOps.BUFFER, src=(UPat.var("to_store"),), name="buf"), append_buffer),
])

# *********

def append_kernel(k:List[UOp], base:UOp):
  k.append(base.sink())
  return base.replace(src=())
break_sched = PatternMatcher([
  (UPat(UOps.BUFFER, src=(UPat(),), name="base"), append_kernel),
])

# *********

pm_remove_buffer = PatternMatcher([(UPat(UOps.VIEW, src=(UPat(UOps.BUFFER, src=(UPat.var('x'),)),)), lambda x: x), ])
def add_buffer(to_realize:Tuple[Dict[UOp, Optional[UOp]], Dict[UOp, UOp]], x:UOp):
  #print(x.op, x.arg)
  # TODO: ugh, this is the worst way to do this
  with Context(TRACK_MATCH_STATS=0): x_bl = graph_rewrite(x, pm_remove_buffer)
  if to_realize.get(x_bl, True) is None:
    print(len(to_realize), "HIT", sum((x is not None) for x in to_realize.values()))
    to_realize[x_bl] = ret = UOp.new_buffer(x.dtype, x.device, x.size, (x,))
    return ret.reshape(x.shape)
  return None
pm_add_buffer = PatternMatcher([(UPat(tuple(UOps), name="x"), add_buffer), ])

@track_rewrites
def _schedule_rewrite(sink:UOp) -> List[ScheduleItem]:
  sink = graph_rewrite(sink, pm_merge_views_and_consts)
  to_realize: Dict[UOp, UOp] = {x.base:None for x in sink.src}
  # mark buffers to be realized
  for p in sink.sparents:
    if p.op is UOps.COPY:
      to_realize[p.src[0]] = None
      to_realize[p] = None
    if p.op is UOps.CONTIGUOUS:
      to_realize[p] = None
    # very simple rule
    if p.op is UOps.REDUCE_AXIS:
      to_realize[p] = None
  sink = graph_rewrite(sink, pm_add_buffer, to_realize)
  sink = graph_rewrite(sink, pm_push_views)
  graph_rewrite(sink, break_sched, sched:=[])
  ret = []
  for s in sched:
    ast = graph_rewrite(s, enumerate_bufs, bufs:=[])
    ret.append(ScheduleItem(ast, bufs))
  return ret

def create_schedule_with_vars(sched:List[UOp]) -> Tuple[List[ScheduleItem], Dict[Variable, int]]:
  sink = UOp.sink(*[x.base for x in sched])
  sched = _schedule_rewrite(sink)
  print(len(sched))
  return sched, {}
