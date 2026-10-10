"""Validate the actual effective config before any training; reuse original recorder."""
from pathlib import Path
import json
import os
import sys

work = Path(os.environ['MF_RUN_DIR'])
expected = json.loads((work / 'seed-check-provenance.json').read_text())['expected_effective_config']
sys.path.insert(0, str(work.parents[1] / 'meanflow_table1e_runbook'))
import audit_run as audit

original_install = audit.install
def guarded_install(recorder, check_only=False):
    original_install(recorder, check_only=check_only)
    if not check_only:
        import train
        recorded_train = train.train_and_evaluate
        def checked_train(config, workdir):
            if config.to_dict() != expected:
                raise RuntimeError('Effective config differs beyond training.seed; zero training steps executed.')
            print('EFFECTIVE CONFIG VERIFIED: seed=43, all other settings match this p in the primary run.', flush=True)
            return recorded_train(config, workdir)
        train.train_and_evaluate = checked_train
audit.install = guarded_install
audit.main()
