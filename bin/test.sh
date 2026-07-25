#!/bin/sh

# Usage examples:
#   bin/test.sh
#   bin/test.sh test_runner.py
#   bin/test.sh test_runner.py::test_scalar_round_trip

set -e

dir=$(dirname "$0")
cd "$dir/.."

if [ $# -gt 0 ]
then
  uv run python -m pytest -v -p no:faulthandler $@
else
  uv run python -m pytest -v -p no:faulthandler tests
fi
