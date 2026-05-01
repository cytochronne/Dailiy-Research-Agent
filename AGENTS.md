# AGENTS.md

## Development Environment

- Always use the dedicated conda environment `daily-arxiv-agent` for this repository.
- Run Python commands with `conda run -n daily-arxiv-agent ...` or activate the environment first with `conda activate daily-arxiv-agent`.
- Do not run project development, dependency installation, or tests from conda `base` unless the user explicitly asks for it.
- Expected interpreter: `/home/cytochrome/pan1/miniconda3/envs/daily-arxiv-agent/bin/python`.
- Standard test command: `conda run -n daily-arxiv-agent python -m pytest -q`.
