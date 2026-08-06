# Local LeRobot snapshot and reproduction notes

This file describes the workstation state captured on 2026-08-06. The goal is
to reproduce the local experiments, not to update this checkout to the latest
upstream LeRobot release.

## Git snapshot

- Fork: `https://github.com/May570/lerobot.git`
- Snapshot branch: `backup/local-lerobot-20260806`
- Snapshot base: `kalman_new` at `f12365b07d1972cbef51846cb5914697be8fee29`
- Upstream merge base recorded locally: `0db5f66ddae6afd62454c418771286fa64dbea20`
  (2026-03-11)
- At backup time, the local feature history was 25 commits ahead of that merge
  base. The newer `origin/main` history was deliberately not merged or rebased.

The snapshot additionally saves three modified realtime evaluation scripts,
three previously untracked dyn-mini observation diagnostic scripts, and the
small experiment drivers that had been stored under the ignored `outputs/`
tree.

## Required workspace layout

Several scripts derive paths from a sibling checkout. Keep both repositories
under one workspace directory:

```text
<workspace>/
  lerobot/
  LIBERO/
```

Use the corresponding LIBERO snapshot:

```text
repository: https://github.com/May570/LIBERO.git
branch:     backup/local-libero-20260806
```

The dyn-mini dataset is not stored in either Git repository. On the old
workstation it was available at:

```text
<workspace>/LIBERO/libero_dyn_mini/datasets/libero_dyn_mini_balanced500_scripted_v2
```

Its core Parquet data was compared with the server copy under
`/share/project/wujiling/datasets/libero-dyn-mini` and judged to be the same
dataset. Restore the dataset separately or point `--dataset.root` at its new
location.

Model checkpoints are also external to Git and are being backed up separately
to ModelScope. Most historical run scripts still contain the old absolute
checkpoint paths; replace them or pass the corresponding environment variable
before running on a new machine.

## Python environment

The active shared environment was:

```text
conda environment: lerobot_py312
path:              /home/admin123/anaconda3/envs/lerobot_py312
Python:            3.12.9
LeRobot:           0.5.1, editable from this checkout
PyTorch:           2.10.0+cu128
CUDA runtime:      12.8
hf-libero:         0.1.3
robosuite:         1.4.0
```

This was not a repository-specific clean environment; it was shared with other
local work. `LOCAL_ENVIRONMENT_CONDA_HISTORY.yml` records the explicitly
requested Conda packages, and `LOCAL_ENVIRONMENT_PIP_FREEZE.txt` records all
packages currently installed in it. Prefer installing this checkout first and
then use the package list as a version reference rather than assuming every
listed package is required:

```bash
conda env create -f LOCAL_ENVIRONMENT_CONDA_HISTORY.yml
conda activate lerobot_py312
python -m pip install -e ".[libero]"
```

## LIBERO local config

`.libero/config.yaml` is normally ignored because it contains machine-local
paths. Copy `.libero/config.example.yaml` to `.libero/config.yaml` and replace
`<workspace>` and `<dataset-root>` for the new machine. The configuration points
standard LIBERO assets, BDDL files and init states to the sibling `LIBERO`
checkout.

Dyn-mini scripts additionally use:

```bash
export LIBERO_CONFIG_PATH="<workspace>/LIBERO/libero_dyn_mini/config"
export PYTHONPATH="<workspace>/LIBERO/libero_dyn_mini/py:<workspace>/lerobot/src"
```

## Saved experiment code under outputs

`outputs/` remains ignored as a result directory. A small, explicit set of
files is nevertheless tracked with Git at their original paths:

- all local `.py`, `.sh`, and `.md` experiment drivers found there at backup
  time;
- `outputs/eval7/fixed_starts/`, containing the fixed 100-episode start cache
  and seen/unseen split;
- `outputs/eval9/fixed_starts/`, containing the dataset-aligned 200-episode
  plan and start-state cache.

These tracked files recreate the experiment entry points without committing
videos, logs, plots, tensors, Parquet predictions or metric tables. Some scripts
record absolute paths from the old machine. They are retained verbatim as an
experiment record; update the workspace, dataset and checkpoint path variables
when rerunning them.

The main groups are:

- `eval2`-`eval4`: early Kalman and dyn-mini evaluation runs;
- `eval5`-`eval7`: future-condition, delay, gate, label and mask ablations;
- `eval8`: action-step, observation-history and zero-state comparisons;
- `eval9`-`eval12`: dataset-vs-online observation and counterfactual probes;
- `eval13`: PI0.5 realtime evaluation;
- `eval14`: ball-position-history evaluation;
- `compare`, `ballpos_oracle_vs_kalman`, and `state_oracle_vs_kalman`:
  diagnostic comparison and plotting code.

## Results excluded from Git

At backup time, `outputs/` occupied about 5.8 GiB and mainly contained 32,300
MP4 videos, 10,673 PNG images, JSON/JSONL summaries, logs, CSV tables and 99
small diagnostic `.pt` tensors. It did not contain training checkpoints.

The largest groups were approximately:

```text
outputs/eval7                         2.1 GiB
outputs/eval9                         1.6 GiB
outputs/eval6                         617 MiB
outputs/eval5                         324 MiB
outputs/compare                       286 MiB
outputs/state_oracle_vs_kalman        260 MiB
outputs/eval12                        171 MiB
outputs/eval8                         154 MiB
outputs/ballpos_oracle_vs_kalman       98 MiB
```

These artifacts require a separate archive if the historical results themselves
must be retained. The fixed-start inputs and experiment code do not depend on
that archive.

## Local-only items that need no backup

- `.venv/`: a small local placeholder; the actual environment is Conda
  `lerobot_py312`.
- `.vscode/`: editor settings only.
- `__pycache__/`, `.pytest_cache/`, logs and other caches: generated files.
- `outputs/eval13/policy_override/`: absolute symlinks into external model
  checkpoints; restore the model from its separate backup instead.
