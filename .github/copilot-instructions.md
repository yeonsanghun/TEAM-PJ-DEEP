<!-- Copilot / AI agent instructions for team-pj-deep -->
# Quick orientation

- **Project type**: small ML/analysis project with interactive notebooks and a tiny runner script. Key items: [main.py](main.py), [multi_label.ipynb](multi_label.ipynb), [all_code.ipynb](all_code.ipynb), [pyproject.toml](pyproject.toml), and the [data/](data/) folder.
- **Python**: project targets Python >=3.12 (see [pyproject.toml](pyproject.toml)). A `.venv/` exists in the workspace — prefer using it for local runs.

# What matters to the agent (actionable rules)

- Inspect notebooks first for experimental code paths. Notebooks contain the model training/analysis workflow; reproduce any changes as small scripts when asked.
- Non-notebook code lives in `main.py` and plain Python modules. Avoid large edits in notebooks without the user's approval; if you change a notebook, clear outputs and keep diffs minimal.
- Do not add or modify large data files under [data/](data/) — mention storage or download steps instead.

# Build / run / debug

- Quick run: `python main.py` runs the repo's main entrypoint (minimal greeting). Use the repository Python (`.venv`) to run commands on Windows (cmd):

  - `\.venv\Scripts\activate` then `python main.py`
  - PowerShell: `\.venv\Scripts\Activate.ps1` then `python main.py`

- Dependency list: see [pyproject.toml](pyproject.toml). There is no build backend configured in-tree; ask before changing packaging.

# Patterns & conventions (discoverable)

- Experiments are in notebooks. When converting notebook logic to scripts, keep function boundaries (data loading → preprocessing → model → evaluation) visible and add a small CLI wrapper.
- Data files are expected under [data/](data/). Code references datasets relative to that folder; prefer stable relative paths.
- No tests found in the repo. If you propose adding tests, present small, focused examples (one test per new utility) and get confirmation.

# Integration / external points

- External deps are only in [pyproject.toml](pyproject.toml) (numpy, pandas, scikit-learn, openpyxl). Assume standard PyPI installs.
- No CI/workflow files discovered — do not add automated pipelines without the user's direction.

# Helpful examples for prompts (how to ask the agent)

- "Refactor preprocessing from `multi_label.ipynb` into `src/preprocess.py` and add a simple CLI wrapper."  — agent should produce a small script, unit example, and clear notebook output.
- "Update dependencies in pyproject and pin versions" — show exact lines to change in [pyproject.toml](pyproject.toml) and explain runtime impacts.

# Safety & scope

- Keep edits small and reversible. Prefer adding new files over changing many existing ones.
- When uncertain about experiment results or expected outputs, ask the user instead of guessing.

---
Please review this draft: tell me what additional repository-specific patterns or files I should surface.
