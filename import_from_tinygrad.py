#!/usr/bin/env python3
import pathlib

FILES = ["tensor.py", "function.py", "helpers.py", "dtype.py", "device.py", "multi.py",
         "nn/__init__.py", "nn/datasets.py", "nn/optim.py", "nn/state.py", "ops.py",
         "shape/symbolic.py", "shape/shapetracker.py", "shape/view.py",
         "runtime/ops_clang.py", "renderer/__init__.py", "renderer/cstyle.py"]
src = pathlib.Path("../tinygrad/tinygrad")
dest = pathlib.Path("toonygrad")

for f in FILES:
  print("importing", f)
  rd = open(src/f).read()
  rd = rd.replace("from tinygrad.", "from toonygrad.")
  rd = rd.replace("import tinygrad.", "import toonygrad.")
  (dest/f).parent.mkdir(parents=True, exist_ok=True)
  with open(dest/f, "w") as f:
    f.write(rd)
