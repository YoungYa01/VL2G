import subprocess
from pathlib import Path

subprocess.run([
    "python",
    "/mnt/d/research_vl2g/scripts/goose_vl2g_pipeline.py",
    "evaluate",
    "--out-root",
    "/mnt/d/research_vl2g/goose_train_500"
], check=True)
