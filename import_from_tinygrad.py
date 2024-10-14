#!/usr/bin/env python3
import os, pathlib

FILES = ["tensor.py", "function.py", "helpers.py", "dtype.py", "device.py", "multi.py",
         "nn/__init__.py", "nn/datasets.py", "nn/optim.py", "nn/state.py", "ops.py",
         "shape/shapetracker.py", "shape/view.py",
         "runtime/ops_clang.py", "runtime/ops_python.py", "runtime/ops_metal.py",
         "renderer/__init__.py", "renderer/cstyle.py",
         "codegen/lowerer.py", "codegen/linearize.py",
         "codegen/uopgraph.py", "codegen/transcendental.py",
         "viz/serve.py", "viz/index.html"]
src = pathlib.Path("../tinygrad/tinygrad")
dest = pathlib.Path("toonygrad")

insert = pathlib.Path("uop_is_lazybuffer.py").read_text()

for f in FILES:
  rd = open(src/f).read()
  rd = rd.replace("from tinygrad.", "from toonygrad.")
  rd = rd.replace("import tinygrad.", "import toonygrad.")
  if f == "ops.py":
    n1 = "@dataclass(frozen=True)\nclass KernelInfo:\n"
    n0,n2 = rd.split(n1)
    rd = n0+insert+n1+n2
  (dest/f).parent.mkdir(parents=True, exist_ok=True)
  if rd == open(dest/f).read(): continue
  if not (dest/f).exists() or int(os.getenv("FORCE", "0")):
    print("importing", f)
    with open(dest/f, "w") as f: f.write(rd)
  else:
    print("skipping", f)
