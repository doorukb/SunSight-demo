"""Shared locations for every experiment.

Importing this module also puts the vendored SunSight toolkit (SunSight_Q3) and Utils on sys.path,
so `from Simulation.projections_util import ...` and `import q4_eva` work from any folder.
"""
import sys
from pathlib import Path

UTILS = Path(__file__).resolve().parent
ROOT = UTILS.parent
DATA = ROOT / "Data"
SUNSIGHT = ROOT / "SunSight_Q3"  # coopersigrist/SunSight at commit 99588d1696c980075a19abf087c669ae5aa22e4e
NEAT_CONFIG = SUNSIGHT / "Models" / "Neat" / "neat-config"
Q4, Q5, Q6 = (ROOT / name for name in ("Q4_EVA_reconstruction", "Q5_softmax_EVA", "Q6_state_EVA"))

for folder in (SUNSIGHT, UTILS):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
