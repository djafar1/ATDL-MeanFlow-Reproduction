#!/usr/bin/env zsh
cd /work/meanflow-table1e || exit 1

if [[ ! -f provenance/pre-training-freeze.txt ]]; then
  venv/bin/python -m pip freeze \
    > provenance/pre-training-freeze.txt || exit 1
fi

venv/bin/python -m pip install \
  --cache-dir cache/pip \
  --constraint provenance/pre-training-freeze.txt \
  --report provenance/training-install-report.json \
  tensorflow-cpu clu PyYAML || exit 1

venv/bin/python -m pip check || exit 1
venv/bin/python -m pip freeze \
  > provenance/training-packages.txt || exit 1

print "TRAINING DEPENDENCIES INSTALLED"
