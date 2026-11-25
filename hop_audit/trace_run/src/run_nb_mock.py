"""Execute the built notebook's cells in order (mock model), handling the %%writefile cell."""
import json, os, sys
from pathlib import Path
nb = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
g = {"__name__": "__main__"}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if src.startswith("%%writefile"):
        first, body = src.split("\n", 1)
        Path(os.environ["TRACE_WORKER_PATH"]).write_text(body, encoding="utf-8")
        print(f"[harness] cell {i}: wrote worker ({len(body)} chars)")
        continue
    exec(compile(src, f"cell{i}", "exec"), g)
