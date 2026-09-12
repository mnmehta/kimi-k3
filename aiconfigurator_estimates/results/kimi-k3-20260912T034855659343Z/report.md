# Kimi-K3 AIC estimate versus H200 ground truth

This report compares AIConfigurator estimates with the published vLLM-bench artifacts for the four 16-GPU strategies. The 32-GPU P/D strategy is intentionally omitted. Ground truth is the artifact's `mean_ttft_ms`, `mean_tpot_ms`, `mean_itl_ms`, and `output_throughput`; AIC uses the matching 1000-token/1000-token `agg` estimate.

Ground-truth repository: `/private/tmp/kimi-k3-ground-truth`
AIC backend: vLLM 0.24.0; system: H200 SXM; model: moonshotai/Kimi-K3.
For PP2 and DP2, AIC's effective concurrency is `batch_size × PP × attention-DP`; odd target concurrencies therefore use the next representable batch and are shown explicitly.

| Strategy | Target C | AIC C | C match | TTFT GT/AIC (ms) | TTFT error | TPOT GT/AIC (ms) | TPOT error | ITL GT (ms) | Output tok/s GT/AIC | Throughput error |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| TP16 | 1 | 1 | yes | 216.1 / 415.3 | +92.1% | 25.2 / 30.8 | +22.5% | 25.2 | 39.4 / 32.0 | -18.9% |
| TP16 | 2 | 2 | yes | 339.5 / 467.4 | +37.7% | 25.8 / 32.9 | +27.6% | 25.8 | 76.6 / 60.0 | -21.8% |
| TP16 | 4 | 4 | yes | 543.5 / 519.9 | -4.3% | 27.1 / 37.3 | +37.5% | 27.1 | 144.6 / 105.7 | -26.9% |
| TP16 | 8 | 8 | yes | 774.0 / 573.2 | -26.0% | 30.8 / 43.9 | +42.6% | 30.8 | 253.6 / 179.8 | -29.1% |
| TP16 | 16 | 16 | yes | 1199.4 / 628.0 | -47.6% | 74.9 / 56.6 | -24.5% | 74.9 | 210.3 / 279.7 | +33.0% |
| TP16 | 32 | 32 | yes | 10987.8 / 686.3 | -93.8% | 53.4 / 70.9 | +32.8% | 53.4 | 433.1 / 446.8 | +3.2% |
| TP16 | 64 | 64 | yes | 66528.7 / 750.9 | -98.9% | 57.1 / 95.4 | +67.0% | 57.1 | 402.2 / 665.5 | +65.5% |
| TP16 | 128 | 128 | yes | 175716.4 / 829.9 | -99.5% | 57.6 / 130.2 | +125.9% | 57.6 | 406.5 / 977.1 | +140.4% |
| TP16 | 256 | — | — | estimate failed | — | — | — | — | — | — |
| TP16 | 512 | — | — | estimate failed | — | — | — | — | — | — |
| TEP16 | 1 | 1 | yes | 264.6 / 430.8 | +62.8% | 28.2 / 33.3 | +18.2% | 28.2 | 35.2 / 29.6 | -15.8% |
| TEP16 | 2 | 2 | yes | 393.7 / 484.9 | +23.2% | 28.7 / 33.9 | +18.2% | 28.7 | 68.9 / 58.2 | -15.5% |
| TEP16 | 4 | 4 | yes | 669.0 / 539.3 | -19.4% | 30.4 / 37.6 | +23.7% | 30.4 | 129.0 / 105.0 | -18.6% |
| TEP16 | 8 | 8 | yes | 984.5 / 594.4 | -39.6% | 34.6 / 46.1 | +33.5% | 34.6 | 225.3 / 171.2 | -24.0% |
| TEP16 | 16 | 16 | yes | 1512.7 / 650.9 | -57.0% | 79.0 / 58.4 | -26.0% | 79.0 | 199.0 / 270.9 | +36.2% |
| TEP16 | 32 | 32 | yes | 15007.1 / 710.7 | -95.3% | 56.7 / 73.5 | +29.6% | 56.7 | 362.0 / 431.1 | +19.1% |
| TEP16 | 64 | 64 | yes | 78278.8 / 776.9 | -99.0% | 58.4 / 105.9 | +81.2% | 58.4 | 366.8 / 599.9 | +63.5% |
| TEP16 | 128 | 128 | yes | 203489.9 / 857.2 | -99.6% | 58.8 / 149.7 | +154.7% | 58.8 | 373.9 / 850.0 | +127.3% |
| TEP16 | 256 | — | — | estimate failed | — | — | — | — | — | — |
| TEP16 | 512 | — | — | estimate failed | — | — | — | — | — | — |
| TP8xPP2 | 1 | 2 | NO | 244.4 / 349.5 | +43.0% | 17.2 / 20.9 | +21.7% | 17.2 | 57.5 / 94.2 | +63.8% |
| TP8xPP2 | 2 | 2 | yes | 399.5 / 349.5 | -12.5% | 17.7 / 20.9 | +17.9% | 17.7 | 110.5 / 94.2 | -14.8% |
| TP8xPP2 | 4 | 4 | yes | 624.3 / 393.4 | -37.0% | 20.2 / 23.7 | +16.9% | 20.2 | 192.0 / 166.3 | -13.4% |
| TP8xPP2 | 8 | 8 | yes | 897.8 / 437.5 | -51.3% | 22.7 / 26.9 | +18.4% | 22.7 | 339.3 / 292.9 | -13.7% |
| TP8xPP2 | 16 | 16 | yes | 1211.1 / 482.2 | -60.2% | 26.5 / 33.6 | +26.8% | 26.5 | 577.2 / 469.0 | -18.7% |
| TP8xPP2 | 32 | 32 | yes | 1701.1 / 527.9 | -69.0% | 34.3 / 46.0 | +34.0% | 34.3 | 888.8 / 688.0 | -22.6% |
| TP8xPP2 | 64 | 64 | yes | 2896.9 / 577.2 | -80.1% | 49.2 / 59.4 | +20.7% | 49.2 | 1228.8 / 1067.5 | -13.1% |
| TP8xPP2 | 128 | 128 | yes | 4603.5 / 629.2 | -86.3% | 71.0 / 85.7 | +20.6% | 71.0 | 1690.6 / 1482.9 | -12.3% |
| TP8xPP2 | 256 | 256 | yes | 8132.0 / 690.4 | -91.5% | 108.6 / 112.0 | +3.1% | 108.6 | 2187.4 / 2272.1 | +3.9% |
| TP8xPP2 | 512 | — | — | estimate failed | — | — | — | — | — | — |
| TP8xDP2 | 1 | 2 | NO | 888.0 / 536.0 | -39.6% | 23.1 / 51.6 | +123.8% | 23.1 | 41.8 / 38.3 | -8.2% |
| TP8xDP2 | 2 | 2 | yes | 5719.2 / 536.0 | -90.6% | 23.1 / 51.6 | +123.7% | 23.1 | 69.5 / 38.3 | -44.8% |
| TP8xDP2 | 4 | 4 | yes | 21927.2 / 603.3 | -97.2% | 23.1 / 57.2 | +148.0% | 23.1 | 83.4 / 69.2 | -17.0% |
| TP8xDP2 | 8 | 8 | yes | 54832.2 / 671.1 | -98.8% | 23.1 / 64.2 | +178.3% | 23.1 | 83.4 / 123.4 | +47.9% |
| TP8xDP2 | 16 | 16 | yes | 124475.8 / 739.8 | -99.4% | 22.7 / 72.9 | +220.4% | 22.7 | 85.0 / 217.4 | +155.9% |

## Mean absolute percentage error

| Strategy | TTFT | TPOT | Output throughput | Points compared |
|---|---:|---:|---:|---:|
| TP16 | 62.5% | 47.6% | 42.3% | 8 |
| TEP16 | 62.0% | 48.1% | 40.0% | 8 |
| TP8xPP2 | 59.0% | 20.0% | 19.6% | 9 |
| TP8xDP2 | 85.1% | 158.8% | 54.8% | 5 |

## Interpretation

AIC's single-point `agg` model represents steady-state scheduler behavior differently from the benchmark's finite burst sweep. TTFT is therefore most comparable at low concurrency; at high concurrency, benchmark queueing and KV-cache limits can dominate. TPOT/ITL is the cleaner decode comparison, while ITL is reported as the ground-truth companion to TPOT.

The complete machine-readable comparison is in `comparison.json`; raw AIC rows and per-operation source tags are in `results.jsonl`.
