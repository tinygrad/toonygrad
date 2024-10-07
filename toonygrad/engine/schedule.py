from typing import List
from toonygrad.ops import UOp, graph_rewrite, PatternMatcher

class ScheduleItem:
  pass

def create_schedule_with_vars(sched:List[UOp]):
  # TODO: should the input be a SINK?
  sink = UOp.sink(*sched)
  sink = graph_rewrite(sink, PatternMatcher([]))
  print(sink)
  return [], {}
