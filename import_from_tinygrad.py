#!/usr/bin/env python3
import os, pathlib

FILES = ["tensor.py", "function.py", "helpers.py", "dtype.py", "device.py", "multi.py",
         "nn/__init__.py", "nn/datasets.py", "nn/optim.py", "nn/state.py", "ops.py",
         "shape/symbolic.py", "shape/shapetracker.py", "shape/view.py",
         "runtime/ops_clang.py", "runtime/ops_python.py",
         "renderer/__init__.py", "renderer/cstyle.py",
         "codegen/lowerer.py", "codegen/linearize.py",
         "viz/serve.py", "viz/index.html", "viz/favicon.svg"]
src = pathlib.Path("../tinygrad/tinygrad")
dest = pathlib.Path("toonygrad")

for f in FILES:
  rd = open(src/f).read()
  rd = rd.replace("from tinygrad.", "from toonygrad.")
  rd = rd.replace("import tinygrad.", "import toonygrad.")
  (dest/f).parent.mkdir(parents=True, exist_ok=True)
  if not (dest/f).exists() or int(os.getenv("FORCE", "0")):
    print("importing", f)
    with open(dest/f, "w") as f: f.write(rd)
  else:
    cmp = open(dest/f).read()
    if rd != cmp: print("skipping", f)
