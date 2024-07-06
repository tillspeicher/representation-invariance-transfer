#!/bin/bash
set -e

# Add the libraries to the PYTHONPATH manually, since they are otherwise
# not available with the current pip-based installation format
# export PYTHONPATH=$PYTHONPATH:/home/exp/src/lib_project:/home/exp/src/lib_llm

exec "$@"
