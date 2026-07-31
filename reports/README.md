# Reports

Quarto sources for publishing harness results.

## Concurrency sweep (1000/1000)

```bash
# Python deps (Jupyter engine + Plotly)
python3 -m venv reports/.venv
reports/.venv/bin/pip install -r reports/requirements.txt
reports/.venv/bin/python -m ipykernel install --user \
  --name=kimi-k3-reports --display-name="Python (kimi-k3 reports)"

# Quarto CLI — brew cask needs sudo here; a user-local binary is under reports/bin
export PATH="$PWD/reports/bin:$PATH"
export QUARTO_PYTHON="$PWD/reports/.venv/bin/python"

quarto render reports/concurrency-sweep-1000-1000.qmd --to html
# writes: reports/concurrency-sweep-1000-1000.html (self-contained, ~30MB)
```

Source: `reports/concurrency-sweep-1000-1000.qmd`  

Data:

- Real weights (primary): `bench-results/conc-sweep-real-1000-1000/`
- Full HF dummy: `bench-results/conc-sweep-full-1000-1000/`
- Shallow 12L dummy: `bench-results/conc-sweep-1000-1000/`
