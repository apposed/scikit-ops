"""Put src/ on sys.path, so `skop` imports from source.

This mirrors what the Runner does for worker processes, where the same
directory is injected through the service init script.
"""

import sys
from pathlib import Path

src = str(Path(__file__).resolve().parent / "src")
if src not in sys.path:
    sys.path.insert(0, src)
