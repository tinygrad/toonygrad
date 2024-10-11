from typing import List
from toonygrad.codegen.lowerer import rewrite_shapetracker_with_index
from toonygrad.ops import track_rewrites, UOp, UOps, BinaryOps, identity_element, PatternMatcher, UPat, graph_rewrite, symbolic_flat
from toonygrad.device import Device
from toonygrad.dtype import dtypes
from toonygrad.helpers import partition
from toonygrad.engine.schedule import ScheduleItem
from toonygrad.codegen.linearize import linearize_uop
from toonygrad.codegen.uopgraph import full_graph_rewrite
from toonygrad.renderer import Renderer
from toonygrad.codegen.kernel import Kernel

@track_rewrites
def _rewrite_kernel(k:Kernel, sink:UOp, opts:Renderer) -> UOp:
  sink = rewrite_shapetracker_with_index(sink, opts)
  sink = full_graph_rewrite(sink, opts)
  return sink

def run_schedule(schedule:List[ScheduleItem], var_vals, do_update_stats=False):
  for i,si in enumerate(schedule):
    dev = Device[si.ast.device]
    sink = _rewrite_kernel(Kernel(f"kernel_{i}"), si.ast, dev.renderer)
    src = dev.renderer.render("kernel", linearize_uop(sink))
    print(src)
    lib = dev.compiler.compile_cached(src)
    prg = dev.runtime("kernel", lib)
    prg(*[x.ensure_allocated()._buf for x in si.bufs])

def memory_planner(x): return x
