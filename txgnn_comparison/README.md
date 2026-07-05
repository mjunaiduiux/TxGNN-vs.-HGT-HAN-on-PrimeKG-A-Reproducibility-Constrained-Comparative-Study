# TxGNN vs HGT/HAN Comparative Study — Working Notes

Tracks deviations from `TxGNN_PrimeKG_Implementation_Plan_v2.md` and current phase status.

## Environment deviation (Phase 0)

The plan assumes a local conda environment. The local machine was checked and found to have:
- ~4 GB total RAM (plan recommends 16 GB minimum)
- **2.4 GB free disk out of 119 GB total — 99% full** (plan needs ~10 GB+ for data/checkpoints, plus several GB for PyTorch/DGL/PyG)
- No CUDA GPU, no usable local Python (only a Windows Store stub on PATH)

Given these constraints, all training and data work runs on **Google Colab** (GPU runtime) instead of locally. This machine only stores code, notebooks, and the small results/JSON files produced by Colab runs.

A second deviation: the plan pins Python 3.8 + DGL 0.5.2 exactly, and assumes the `TxGNN` and `DGL` pip packages install cleanly.
- Forcing Python 3.8 via `condacolab` broke immediately (`install_miniconda()` doesn't take a `python_version` argument).
- Dropping that, the `TxGNN` pip package itself fails `setup.py egg_info` on Colab's current Python (3.12) — it depends on `distutils`, removed from the stdlib in 3.12. The package hasn't been updated since ~2022, so this isn't a transient issue.

**Resolution:** `TxGNN` and `DGL` are both dropped as dependencies. Phase 2 implements TxGNN's published architecture — relational GNN pretraining encoder + metric-learning decoder for indication/contraindication link prediction, two-phase pretrain→fine-tune — directly in PyTorch Geometric, using the paper and the `mims-harvard/TxGNN` source as the spec rather than the broken pip package. Every metric reported from this will be labelled "TxGNN-architecture reimplementation (PyG), this study" rather than implying the original authors' exact checkpoint, since we no longer load their pretrained weights (those were tied to the old DGL graph format). This is the most significant deviation in the project and will be stated explicitly in the website and report, not just here.

## Folder structure

```
txgnn_comparison/
  notebooks/          Colab notebooks, one per phase
  baselines/          HGT/HAN baseline source (mirrored into Phase 3 notebook)
  results/            Results JSON files copied back from Colab after each run
  data/, checkpoints/ Empty locally — real data/checkpoints live on Google Drive, mounted from Colab
```

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 0 — Environment setup | Done | Ran in Colab: `CUDA available: True`, PyTorch Geometric 2.8.0, PyTorch 2.10.0+cu128 |
| 1 — Data acquisition | Done | Real PrimeKG counts confirmed: 17,080 disease nodes, 7,957 drug nodes, 129,375 total nodes, 8,100,498 total edges. `data_card.json` and `references.json` saved to `results/` |
| 2 — TxGNN fine-tune + eval | Done | Real results in `results/txgnn_pyg_random.json` and `results/txgnn_pyg_zero_shot.json` |
| 3 — HGT/HAN baseline | Done | First run OOM'd on a T4 (full-batch attention over 8.1M edges) — fixed with hidden 64→32, heads 4→2, edges capped at 50k/relation, fp16 autocast. Real results in `results/hgt_*.json` and `results/han_*.json`. Breaks the "identical hidden dim" fairness condition vs Phase 2; flagged in the report's limitations section |
| 4 — Results aggregation | Done | `results/aggregate_results.py` ran successfully locally (real Python 3.13 found at `C:\Users\TOSHIBA\AppData\Local\Programs\Python\Python313\python` — `python3` on PATH is a Windows Store stub, but plain `python` works). Produced valid `results.json` and `comparison_table.json` from the 6 real result JSONs. All models land in a narrow 0.98-0.99 AUPRC/AUROC band — flagged as a finding (random-negative-sampling evaluation isn't very discriminative), not a clean win for either architecture. HGT/HAN ran at reduced scale (T4 OOM fix) vs TxGNN's full-scale run — flagged as a comparability limitation throughout. |
| 5 — React website | Done (build + dev server verified) | `txgnn-website/` (Vite + React + Recharts + react-router-dom), data copied from `txgnn_comparison/results/`. 6 pages (Overview/Background/Models/Results/Discussion/References), PendingBadge for nulls, ResultsChart only renders when no nulls. `npm run build` succeeds, `npm run dev` serves all routes (200). Not yet checked: responsive breakpoints (375/768/1280px), Lighthouse accessibility, external citation link validity — left for Phase 7. |
| 6 — Report | Done | `REPORT.md` at project root. All 6 references verified via WebSearch/WebFetch (2026-06-29) against primary/indexing sources — one open discrepancy noted (TxGNN paper's exact zero-shot improvement % varies between two automated full-text extractions, not reconciled). Case study quote verified against a PMC mirror of the published paper. Appendix contains the real `results.json` content, substituted programmatically (not hand-copied) |
| 7 — Testing | Not started | |

## How to run Phase 0

1. Open `notebooks/phase0_setup_verify.ipynb` in Google Colab (upload it or open from Drive).
2. `Runtime > Change runtime type > GPU`.
3. `Runtime > Run all`. No restart needed.
4. Confirm the verification cell prints `CUDA available: True` and no `FAILED` line for PyTorch Geometric.
5. Paste back whatever the verification cell printed (including any `FAILED` lines) so the actual versions get logged before moving to Phase 1.
