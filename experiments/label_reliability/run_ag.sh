#!/bin/bash
# One AutoGluon-arm task: SLURM_ARRAY_TASK_ID -> (dataset, seed, chunk)
set -euo pipefail
cd /work/dlclarge1/matusd-schaefeb-metafe/meta-surrogate-feature-engineering
PY=/work/dlclarge1/matusd-schaefeb-metafe/venvs/ag12/bin/python
read D S C < <(PYTHONPATH=. $PY -c "
import sys; sys.path.insert(0, 'experiments/label_reliability')
from label_reliability import DATASETS, AG_SEEDS
i, n = $SLURM_ARRAY_TASK_ID, $N_CHUNKS
print(DATASETS[i // (len(AG_SEEDS) * n)], AG_SEEDS[(i // n) % len(AG_SEEDS)], i % n)")
echo "$(date) $(hostname) dataset $D seed $S chunk $C/$N_CHUNKS"
PYTHONPATH=. $PY experiments/label_reliability/label_reliability.py ag --dataset $D --seed $S --chunk $C --n_chunks $N_CHUNKS
echo "$(date) done"
