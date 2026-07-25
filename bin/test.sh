#!/bin/sh

# Usage examples:
#   bin/test.sh
#   bin/test.sh test_runner.py
#   bin/test.sh test_runner.py::test_scalar_round_trip
#
# Op tests whose environment is not built yet are skipped. To run them all,
# installing whatever is missing (slow the first time, and what CI runs):
#
#   bin/test.sh --build-envs
#
# Or pick a slice of them:
#
#   bin/test.sh --build-envs -m env -k stardist

set -e

dir=$(dirname "$0")
cd "$dir/.."

if [ $# -gt 0 ]
then
  uv run python -m pytest -v -p no:faulthandler "$@"
else
  uv run python -m pytest -v -p no:faulthandler tests
fi
