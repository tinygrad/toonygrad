from typing import List
from toonygrad.codegen.lowerer import ast_to_uop
from toonygrad.ops import track_rewrites, UOp, UOps, BinaryOps, identity_element, PatternMatcher, UPat, graph_rewrite, symbolic_flat
from toonygrad.device import Device
from toonygrad.dtype import dtypes
from toonygrad.helpers import partition
from toonygrad.engine.schedule import ScheduleItem
from toonygrad.codegen.linearize import linearize_uop
from toonygrad.renderer import Renderer

acc_number = 0
def do_reduce(root:UOp):
  global acc_number
  reduce_parented, reduce_unparented = partition(root.src[1:], lambda x: x in root.src[0].sparents)
  ret = root.src[0]
  if len(reduce_parented):
    acc = UOp(UOps.DEFINE_ACC, root.dtype,
              (root.const_like(identity_element(root.arg, root.dtype.scalar())),) + tuple(reduce_parented), (acc_number,))
    acc_number += 1
    ret = UOp(UOps.ASSIGN, root.dtype, (acc, acc.alu(root.arg, ret)))
  # for MAX, we can just ignore the unparented
  if root.arg is BinaryOps.ADD:
    for r in reduce_unparented:ret = ret * (r.src[1]-r.src[0]).cast(ret.dtype.scalar()).broadcast(ret.dtype.count)
  return ret

no_pyint = PatternMatcher([(UPat((UOps.CONST, UOps.VCONST, UOps.ALU, UOps.SPECIAL, UOps.RANGE, UOps.EXPAND, UOps.VECTORIZE, UOps.DEFINE_VAR),
  name="x"), lambda x: UOp(x.op, dtypes.int32.vec(x.dtype.count), x.src, x.arg) if x.dtype.scalar() == dtypes.pyint else None)])

just_reduce = PatternMatcher([
  # do reduce
  (UPat(UOps.REDUCE, name="root"), do_reduce),
])

@track_rewrites
def _rewrite_kernel(s:UOp, opts:Renderer) -> UOp:
  sink = ast_to_uop(s, opts)
  sink = graph_rewrite(sink, symbolic_flat+no_pyint+just_reduce)
  return sink

def run_schedule(schedule:List[ScheduleItem], var_vals, do_update_stats=False):
  for si in schedule:
    dev = Device[si.ast.device]
    sink = _rewrite_kernel(si.ast, dev.renderer)
    src = dev.renderer.render("kernel", linearize_uop(sink))
    lib = dev.compiler.compile_cached(src)
    prg = dev.runtime("kernel", lib)
    prg(*[x.ensure_allocated()._buf for x in si.bufs])

def memory_planner(x): return x
