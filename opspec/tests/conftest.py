"""opspec's core needs only the standard library; most tests here use numpy.

Without numpy installed, the tests that need it are not collected, and
test_no_numpy.py still checks the core.
"""

import importlib.util

collect_ignore = []
if importlib.util.find_spec("numpy") is None:
    collect_ignore = [
        "sample_ops.py",
        "sample_results.py",
        "test_opspec.py",
        "test_plan.py",
        "test_wire.py",
    ]
