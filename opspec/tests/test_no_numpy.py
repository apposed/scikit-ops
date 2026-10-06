"""opspec never imports numpy: it describes arrays without computing on them."""

import subprocess
import sys


def test_importing_opspec_does_not_load_numpy():
    # A fresh interpreter, since this one has numpy loaded by other tests.
    code = (
        "import sys\n"
        "import opspec, opspec.op, opspec.plan, opspec.runner, opspec.types\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'numpy')\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
