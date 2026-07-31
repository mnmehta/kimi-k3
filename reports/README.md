# Reports

Quarto sources for publishing harness results.

## Setup

```bash
python3 -m venv reports/.venv
reports/.venv/bin/pip install -r reports/requirements.txt
reports/.venv/bin/python -m ipykernel install --user \
  --name=kimi-k3-reports --display-name="Python (kimi-k3 reports)"

export PATH="$PWD/reports/bin:$PATH"
export QUARTO_PYTHON="$PWD/reports/.venv/bin/python"
```

## TP16 vs TP8×PP2 (primary)

Real-weight comparison only (no dummy / truncated-layer runs):

```bash
quarto render reports/tp16-vs-pp2-1000-1000.qmd --to html
```

Source: `reports/tp16-vs-pp2-1000-1000.qmd`  
Data: `bench-results/conc-sweep-real-1000-1000/`, `bench-results/conc-sweep-pp2-1000-1000/`  
Pages: [mnmehta.github.io/kimi-k3](https://mnmehta.github.io/kimi-k3/) (this report is `index.html`)

## Full harness history (incl. dummy)

```bash
quarto render reports/concurrency-sweep-1000-1000.qmd --to html
```

Also published at `/concurrency-sweep-1000-1000.html` on Pages.
