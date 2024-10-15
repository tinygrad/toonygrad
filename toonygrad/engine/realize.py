from typing import List
from toonygrad.codegen.lowerer import rewrite_shapetracker_with_index
from toonygrad.ops import track_rewrites, UOp, UOps, BinaryOps, identity_element, PatternMatcher, UPat, graph_rewrite, symbolic_flat
from toonygrad.device import Device
from toonygrad.dtype import dtypes
from toonygrad.helpers import partition, dedup, flatten
from toonygrad.engine.schedule import ScheduleItem
from toonygrad.codegen.linearize import linearize_uop
from toonygrad.codegen.uopgraph import full_graph_rewrite
from toonygrad.renderer import Renderer
from toonygrad.codegen.kernel import Kernel

pm_new_lowerer = PatternMatcher([
  (UPat(UOps.REDUCE_AXIS, name="x"),
   lambda x: UOp(UOps.REDUCE, x.dtype, (x.src[0],)+tuple(UOp.range(dtypes.int, 0, sz, ax) for ax,sz in zip(x.arg[1], x.arg[2])), x.arg[0])),
  (UPat(UOps.VALID, src=(UPat(UOps.VIEW),), name="x"), lambda x: x.st_arg.to_indexed_uops()[1]),
  # rewrite LOAD/STORE VIEW to LOAD/STORE with indexed
  (UPat((UOps.LOAD, UOps.STORE), src=(UPat(), UPat(UOps.VIEW)), allow_any_len=True, name="x"),
   lambda x: UOp(x.op, x.dtype, (x.src[0], x.src[1].st.to_indexed_uops()[0]) + x.src[2:]))
])

@track_rewrites
def _rewrite_kernel(k:Kernel, sink:UOp, opts:Renderer) -> UOp:
  #sink = rewrite_shapetracker_with_index(sink, opts)
  sink = graph_rewrite(sink, pm_new_lowerer)
  # TODO: globally group ranges, replace some with specials, replace some with split ranges. this is "OptOps"
  sink = full_graph_rewrite(sink, opts)
  return sink

def run_schedule(schedule:List[ScheduleItem], var_vals, do_update_stats=False):
  for i,si in enumerate(schedule):
    device = si.bufs[0].device
    print(f"*** kernel {i} on {device}")
    if si.ast.op is UOps.COPY:
      # TODO: actually do COPY
      print(si.bufs)
      bufs = [x.ensure_allocated()._buf for x in si.bufs]
      continue
    dev = Device[device]
    sink = _rewrite_kernel(Kernel(f"kernel_{i}"), si.ast, dev.renderer)
    src = dev.renderer.render("fxn", linearize_uop(sink))
    shapes = dedup([x.st.shape for x in si.ast.sparents if x.op is UOps.VIEW])
    print(shapes)
    print(src)
    lib = dev.compiler.compile_cached(src)
    prg = dev.runtime("fxn", lib)
    prg(*[x.ensure_allocated()._buf for x in si.bufs])

def memory_planner(x): return x
