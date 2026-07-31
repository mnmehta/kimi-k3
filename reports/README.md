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

- TP16 real weights: `bench-results/conc-sweep-real-1000-1000/`
- TP8×PP2 real weights: `bench-results/conc-sweep-pp2-1000-1000/`
- PP2 @ max-num-seqs=32 (partial): `bench-results/conc-sweep-pp2-seqs32-1000-1000/`
- Full HF dummy: `bench-results/conc-sweep-full-1000-1000/`
- Shallow 12L dummy: `bench-results/conc-sweep-1000-1000/`
