# Repository Guidelines

## Project Structure & Module Organization
This repository is split into two editable Python packages:
- `source/ame_locomotion/`: Isaac Lab extension code, task configs, terrains, and robot assets for AME locomotion.
- `rsl_rl/`: the local fork of RSL-RL with custom network, runner, and algorithm changes.

Supporting files live at the repo root:
- `scripts/`: training, play, export, and utility entrypoints.
- `run_train.sh` and `run_play.sh`: project-specific launch wrappers.
- `pretrained/`: shipped checkpoints for quick evaluation.
- `doc/`: migration and experiment notes.

## Build, Test, and Development Commands
Use editable installs while developing:
```bash
python -m pip install -e source/ame_locomotion
python -m pip install -e rsl_rl
```

Common run commands:
```bash
bash run_train.sh   # start AME training
bash run_play.sh    # evaluate / visualize a checkpoint
```

For package-level checks in `rsl_rl`, use the repo’s pre-commit flow:
```bash
pre-commit install
pre-commit run --all-files
```

## Coding Style & Naming Conventions
Python code follows the local `rsl_rl` tooling: 120-character lines, `isort` import ordering, and `pyright` basic type checking. Keep imports grouped and use existing `# isort: skip` markers where the codebase already requires them. Prefer descriptive snake_case for functions, variables, and config files; task/config modules usually mirror the robot or environment name, such as `velocity_env_cfg_29dof.py`.

## Testing Guidelines
There is no standalone test suite in the root. Validate changes by running the relevant training or play script and, for shared Python changes, the `pre-commit` checks above. If you touch environment configs or assets, verify the affected task path under `scripts/rsl_rl/train.py` or `scripts/rsl_rl/play.py`.

## Commit & Pull Request Guidelines
Commit history uses short bracketed prefixes, for example `[feat]:S54 AME training` or `Update README.md`. Keep commits focused and mention the affected robot/task in the subject when relevant. Pull requests should explain the change, note any config or checkpoint impacts, and include screenshots or videos for behavior changes in simulation.

## Configuration Notes
Large assets and checkpoints are already tracked under `pretrained/`, `kuavo/`, and `unitree_model/`. Avoid renaming or moving these without updating the referenced task configs and launch scripts.
