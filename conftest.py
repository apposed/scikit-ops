"""Put the repo root on sys.path, so `opkit` and `ops` import from source.

This mirrors what the Runner does for worker processes, where the same
directory is injected via PYTHONPATH.
"""

import sys
from pathlib import Path

root = str(Path(__file__).resolve().parent)
if root not in sys.path:
    sys.path.insert(0, root)
