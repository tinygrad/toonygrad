from toonygrad.codegen.lowerer import ast_to_uop
from toonygrad.ops import track_rewrites, UOp
from toonygrad.device import Device

@track_rewrites
def _rewrite_kernel(s:UOp) -> UOp:
  opts = Device[s.device].renderer
  return ast_to_uop(s, opts)

def run_schedule(schedule, var_vals, do_update_stats=False):
  for s in schedule:
    sink = _rewrite_kernel(s)
    print(sink.op)

def memory_planner(x): return x
