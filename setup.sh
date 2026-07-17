#!/bin/bash
set -e

python -m pip install --upgrade pip

pip install .
pip install copulas==0.14.1 --no-deps
pip install ctgan==0.12.1 --no-deps
pip install deepecho==0.8.1 --no-deps
pip install rdt==1.21.0 --no-deps
pip install sdmetrics==0.28.0 --no-deps
pip install sdv==1.33.0 --no-deps
pip install statsmodels==0.14.6 --no-deps