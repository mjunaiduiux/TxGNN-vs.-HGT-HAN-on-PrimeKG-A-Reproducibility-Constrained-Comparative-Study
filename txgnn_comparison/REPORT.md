# TxGNN vs. HGT/HAN on PrimeKG: A Reproducibility-Constrained Comparative Study

**Author:** [name not specified in this session]
**Date:** 2026-06-29

## Abstract

Drug repurposing models are usually judged on aggregate accuracy, but the harder and more clinically
relevant test is zero-shot prediction for diseases with no approved therapy at all. TxGNN was built for
exactly that case, pretraining a relational graph encoder across a biomedical knowledge graph before
fine-tuning a metric-learning decoder on drug-disease links. This study set out to reproduce TxGNN
against two attention-based heterogeneous GNN baselines, HGT and HAN, on PrimeKG. The original TxGNN
software could not be installed in the working environment (its dependency on an old DGL release breaks
under current Python), so its architecture was reimplemented from the published description in PyTorch
Geometric instead of loading the authors' checkpoint. All three models were trained on identical
random and zero-shot drug-disease splits built for this study. The result was not the clean separation
reported in the original paper: TxGNN's reimplementation, HGT, and HAN all scored between 0.977 and
0.996 AUPRC/AUROC on both splits, a band too narrow to support a confident ranking. Reduced model scale
(forced by a single T4 GPU's memory limit on the HGT/HAN runs) and an evaluation protocol using uniform
random negative sampling are the two most likely explanations, and both are discussed as limitations
rather than treated as settled.

## 1. Introduction

Most of the difficulty in drug repurposing isn't predicting a treatment for a well-studied disease —
it's predicting one for a disease that has never had an approved therapy at all. Of PrimeKG's 17,080
disease nodes, a large majority fall into that category. A model that performs well only on diseases it
has already seen labeled drugs for is not actually solving the repurposing problem; it is solving a much
easier one. TxGNN (Huang et al., 2024) was built around this distinction directly, using a two-phase
training procedure — self-supervised pretraining across the full knowledge graph, then fine-tuning a
decoder for the indication/contraindication task — specifically to handle diseases absent from the
fine-tuning labels.

This study asks whether that two-phase design actually outperforms a more conventional single-stage,
attention-based heterogeneous GNN (HGT, HAN) when run on the same data and the same splits. It is a
narrower, more tractable stand-in for the broader question of whether knowledge-graph-aware,
metric-learning architectures beat generic attention mechanisms — the same self-attention mechanism
that underlies transformer language models, just applied here to a graph instead of a token sequence.

## 2. Background and related work

**PrimeKG.** PrimeKG (Chandak, Huang & Zitnik, 2023) integrates more than 20 biomedical resources into
one heterogeneous graph spanning diseases, drugs, genes/proteins, phenotypes, anatomy, pathways,
biological processes, and exposures. The instance downloaded for this study reports 17,080 disease
nodes and 7,957 drug nodes, matching the published statistics, across 129,375 total nodes and 8,100,498
total edges (see the data card in the Appendix).

**TxGNN.** TxGNN (Huang, Chandak, Wang, Havaldar, Vaid, Leskovec, Nadkarni, Glicksberg, Gehlenborg &
Zitnik, 2024, *Nature Medicine*) frames drug repurposing as link prediction across two relation types,
indication and contraindication, and trains in two phases: self-supervised pretraining over the entire
knowledge graph, then fine-tuning a metric-learning decoder on the labeled task. The paper reports a
large improvement over prior methods under zero-shot evaluation, though the exact magnitude could not
be pinned down with confidence in this session — two different automated readings of the full text
returned different numbers (49.2% / 35.1% AUPRC gain for indications/contraindications in one reading,
19.0% / 23.9% in another), and that discrepancy is left open here rather than resolved by picking
whichever number sounds better.

**HGT.** Heterogeneous Graph Transformer (Hu, Dong, Wang & Sun, 2020, WWW '20) generalizes transformer-style
multi-head attention to heterogeneous graphs, learning separate attention parameters per
node-type/edge-type combination. It has no built-in pretrain phase; the encoder is trained directly on
whatever supervised task it is pointed at.

**HAN.** Heterogeneous Graph Attention Network (Wang, Ji, Shi, Wang, Cui, Yu & Ye, 2019, WWW '19) takes
a metapath-based approach, computing attention over node-level neighbors within a metapath and then over
the metapaths themselves. Like HGT, it is single-stage and was used as a baseline in TxGNN's original
evaluation.

**HGTDR.** HGTDR (*Bioinformatics*, 2024, DOI 10.1093/bioinformatics/btae349) applies HGT specifically to
drug repurposing on PrimeKG, augmenting node features with BioBERT/ChemBERTa embeddings. It is the
closest published comparable to this study's HGT baseline; this study does not reuse its code, building
HGT independently via PyTorch Geometric's `HGTConv` instead.

**PyTorch Geometric.** PyG (Fey & Lenssen, 2019) is the library underlying every model trained in this
study — the TxGNN reimplementation, HGT, and HAN all share its `HeteroData` graph representation and
message-passing primitives, which is itself part of why a fair architectural comparison was feasible at
all without three separate codebases.

## 3. Case study from the original paper

The original TxGNN paper includes case studies where the model recommended drugs that were not obvious
from disease taxonomy alone. One documented case (verified against a PubMed Central mirror of the
published article, PMC11326339, retrieved 2026-06-29) involves Kleefstra syndrome, a neurodevelopmental
disorder with speech delay and autism-spectrum features:

> "On querying the TXGNN Predictor, zolpidem was recommended as the number one drug repurposing
> candidate."

Zolpidem is ordinarily a sedative — recommending it for a neurodevelopmental disorder looks wrong on its
face. The paper's explainer module reportedly surfaced multi-hop graph paths suggesting a paradoxical
stimulative effect on prefrontal cortex function in some neurological conditions, which is offered as the
mechanistic rationale rather than the model arbitrarily picking a sedative.

This case study matters for this study's results in a specific way: it is exactly the kind of
counterintuitive, mechanism-dependent prediction that a metric-learning decoder reading multi-hop graph
structure is supposed to be good at, and that a baseline trained only to fit observed indication/
contraindication edges is not obviously built to surface. None of the three models trained here were
evaluated on this specific case (Kleefstra syndrome was not isolated as a held-out test case in either
split), so this study cannot confirm or contradict that this study's TxGNN reimplementation reproduces
that specific prediction. That is a limitation, not a finding, and is listed again in Section 8.

## 4. Methods

**Environment.** Training ran on a remote Google Colab GPU runtime (NVIDIA T4, ~14.5-14.6 GB peak memory
observed), accessed via VS Code's Colab extension; the local machine (Windows, ~4 GB RAM, frequently
near disk-full) ran only the lightweight aggregation script. Python on Colab is 3.12; the original
TxGNN pip package depends on `distutils`, removed from the standard library in 3.12, and fails to
install (`setup.py egg_info` error). DGL and the TxGNN package were dropped entirely; PyTorch Geometric
covers both the TxGNN reimplementation and the HGT/HAN baselines.

**Data.** PrimeKG was downloaded directly from Harvard Dataverse (DOI 10.7910/DVN/IXA7BM) via its file
API rather than a hardcoded file ID. Indication and contraindication relation names were located by
substring match on the live data (`'indicat'` / `'contraindicat'`) rather than assumed in advance.

**TxGNN reimplementation.** A relational encoder (2-layer `HeteroConv` with `SAGEConv` per edge type,
hidden dimension 64, learnable per-node-type embeddings as input features since PrimeKG ships none) was
pretrained for 50 epochs via link-prediction loss sampled across all non-target relation types, then
fine-tuned for 200 epochs on indication and contraindication training pairs with a dot-product decoder.
This is a documented simplification of the published architecture: the decoder is dot-product rather
than TxGNN's published metric-learning module, and the encoder uses mean-aggregation (SAGEConv) rather
than the original's specific message-passing design. The dot-product decoder was deliberately kept
identical to the HGT/HAN baselines' decoder so the comparison isolates the encoder's training paradigm
rather than decoder differences.

**HGT/HAN baselines.** Both use PyTorch Geometric's `HGTConv` and `HANConv` respectively, single-stage
supervised training directly on the same indication/contraindication train pairs, with the same
dot-product decoder. The first run out-of-memory'd a T4 GPU doing full-batch attention over PrimeKG's
8.1 million edges. The fix: hidden dimension reduced 64→32, attention heads reduced 4→2, each relation
type capped at 50,000 edges (randomly sampled, seed 42), 100 epochs instead of 200, and mixed-precision
(fp16) forward passes. This means HGT/HAN ran at a smaller scale than the TxGNN reimplementation —
acknowledged directly as a limitation in Section 8, not hidden in a footnote.

**Splits.** Built for this study, not the original TxGNN splits (the original used `TxData`, part of the
dropped package). Random split: 70/15/15 train/valid/test over indication and contraindication pairs
independently, seed 42. Zero-shot split: 20% of diseases with at least one indication/contraindication
edge are selected at random (seed 42) and *all* their edges moved to the test set, so the model never
sees a labeled example for those diseases during training; remaining edges split 85/15 train/valid.
Both TxGNN and HGT/HAN were evaluated against the identical split arrays (saved as `splits.npz` after
the TxGNN run, loaded by the HGT/HAN notebook with an assertion that node counts match).

**Metrics.** AUPRC (`sklearn.metrics.average_precision_score`) and AUROC
(`sklearn.metrics.roc_auc_score`), computed against the test positive pairs plus an equal number of
randomly sampled negative (non-edge) drug-disease pairs.

## 5. Results

All metrics below are this study's own runs — see the Appendix for the complete raw `results.json`.
None are paper-reported numbers; where the original paper's numbers are discussed, they are explicitly
labeled as such in Sections 2 and 6.

### 5.1 Random split

| Model | Indication AUPRC | Indication AUROC | Contraindication AUPRC | Contraindication AUROC | Params | Train time (min) |
|---|---|---|---|---|---|---|
| TxGNN reimplementation (this study) | 0.9911 | 0.9934 | 0.9933 | 0.9956 | 9,105,600 | 1.15 |
| HGT (this study, reduced scale) | 0.9834 | 0.9859 | 0.9945 | 0.9958 | 4,327,100 | 1.54 |
| HAN (this study, reduced scale) | 0.9818 | 0.9861 | 0.9812 | 0.9866 | 4,169,696 | 0.65 |

### 5.2 Zero-shot split

| Model | Indication AUPRC | Indication AUROC | Contraindication AUPRC | Contraindication AUROC |
|---|---|---|---|---|
| TxGNN reimplementation (this study) | 0.9799 | 0.9859 | 0.9832 | 0.9892 |
| HGT (this study, reduced scale) | 0.9769 | 0.9830 | 0.9935 | 0.9947 |
| HAN (this study, reduced scale) | 0.9853 | 0.9883 | 0.9855 | 0.9903 |

The TxGNN reimplementation has the best indication AUPRC on the random split (0.9911 vs. 0.9834 and
0.9818) but not on the zero-shot split, where HAN edges it out on indication AUPRC (0.9853 vs. 0.9799)
and HGT leads on contraindication AUPRC (0.9935 vs. 0.9832). Across both splits, no model is
consistently best on every metric, and every score sits inside a 0.977-0.996 band. That is a narrower
spread than this study expected going in.

## 6. Discussion

**Do attention-based heterogeneous GNNs perform better with a knowledge graph?** Not demonstrably,
based on these runs. The three models land too close together to support a ranking, on either split.
That contradicts the separation the original TxGNN paper reports against HGT/HAN baselines under
zero-shot evaluation — though, as noted in Section 2, this study could not pin down the original paper's
exact improvement percentages with confidence, so "contradicts" should be read as "does not reproduce
the qualitative pattern," not as a precise numerical comparison.

**Is two-phase training the better approach?** The architectural difference was real in this study's
setup — the TxGNN reimplementation pretrained across the full graph before fine-tuning, HGT/HAN did not
— but the results don't show an advantage from it. The most likely reason is the evaluation protocol:
uniform random negative sampling makes most negatives easy to separate from positives regardless of how
well the encoder generalizes, which would compress all three models toward the same high ceiling. A
harder negative-sampling scheme (degree-matched or type-matched negatives) would be a more informative
next experiment than tuning any one model further.

**Why does zero-shot matter?** Independent of this study's inconclusive comparison, the motivating fact
stands: most of PrimeKG's 17,080 diseases have no recorded therapy, and a repurposing model that only
works for diseases with existing labeled drugs is solving a smaller and less useful problem than the one
it's named for.

**Why is attention optional in TxGNN but central in HGT/HAN?** This study's reimplementation used
SAGEConv (mean aggregation, no attention) for the TxGNN encoder, which means the comparison run here is
better described as "two-phase pretrain/fine-tune vs. single-stage training" than as "attention vs. no
attention." Testing the attention question directly would require an attention-based encoder in the
TxGNN reimplementation too, which this study did not build.

## 7. Limitations

- **Not the original TxGNN.** The pip package fails to install on current Python; everything reported
  as "TxGNN" here is a PyTorch Geometric reimplementation of the published architecture, with a simpler
  dot-product decoder and SAGEConv encoder rather than the original's specific design.
- **Unequal scale between TxGNN and HGT/HAN.** HGT and HAN ran with hidden dimension 32 and edges capped
  at 50,000 per relation type (a single T4 GPU OOM'd at the original hidden=64, full-edge-count setup);
  the TxGNN reimplementation ran at hidden=64 with the full 8.1M edges. This breaks the plan's intended
  "identical hidden dimension" fairness condition and is the most likely confound in Section 6's
  inconclusive result.
- **Random negative sampling.** Both training and evaluation use uniformly random non-edges as
  negatives, a known source of inflated AUPRC/AUROC in knowledge-graph link prediction, and the most
  likely reason all three models cluster so tightly.
- **Splits are not the original TxGNN splits.** Built independently for this study (Section 4); results
  are not directly comparable to the original paper's reported numbers even setting aside the model
  differences.
- **No disease-area split.** The plan's optional Phase 2.4 (disease-area holdout) was not run.
- **The Kleefstra syndrome case study (Section 3) was not specifically tested.** This study cannot say
  whether the TxGNN reimplementation reproduces that particular prediction.
- **One unresolved citation discrepancy.** The original paper's exact zero-shot improvement percentage is
  reported differently by two different automated text extractions in this session (Section 2); not
  reconciled against the primary PDF directly.

## 8. Conclusion and future work

This study set out to compare TxGNN's two-phase architecture against HGT/HAN single-stage baselines on
identical PrimeKG splits, and the honest result is that the comparison, as run, doesn't resolve the
question. All three models score within a tight band regardless of architecture or split, most plausibly
because random-negative-sampling evaluation isn't discriminating enough between them, compounded by an
unequal-scale confound forced by GPU memory limits. The two changes most likely to produce a sharper
answer are harder negative sampling and running HGT/HAN at the TxGNN reimplementation's full scale on a
GPU with more memory headroom — both are concrete, runnable next steps rather than open-ended future
work.

## 9. References

1. Chandak, P., Huang, K., & Zitnik, M. (2023). Building a knowledge graph to enable precision medicine. *Scientific Data*, 10, 67. Dataset DOI: 10.7910/DVN/IXA7BM.
2. Huang, K., Chandak, P., Wang, Q., Havaldar, S., Vaid, A., Leskovec, J., Nadkarni, G. N., Glicksberg, B. S., Gehlenborg, N., & Zitnik, M. (2024). A foundation model for clinician-centered drug repurposing. *Nature Medicine*. DOI: 10.1038/s41591-024-03233-x. PMID: 39148855.
3. Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). Heterogeneous Graph Transformer. *Proceedings of The Web Conference 2020 (WWW '20)*, 2704-2710. DOI: 10.1145/3366423.3380027.
4. Wang, X., Ji, H., Shi, C., Wang, B., Cui, P., Yu, P., & Ye, Y. (2019). Heterogeneous Graph Attention Network. *The World Wide Web Conference 2019 (WWW '19)*, 2022-2032. DOI: 10.1145/3308558.3313562.
5. HGTDR: Advancing drug repurposing with heterogeneous graph transformers. (2024). *Bioinformatics*, 40(7), btae349. DOI: 10.1093/bioinformatics/btae349.
6. Fey, M., & Lenssen, J. E. (2019). Fast Graph Representation Learning with PyTorch Geometric. *ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds*. arXiv:1903.02428.

All six references above were verified via web search/fetch against primary or indexing sources
(PubMed, Nature, PMC, ACM DL, Oxford Academic, arXiv) on 2026-06-29 — see `txgnn-website/src/data/references.json`
for per-entry verification notes, including the one open discrepancy noted in Section 2.

## Appendix: Raw experimental results — source of all reported metrics

```json
{
  "last_updated": "2026-06-29",
  "hardware": {
    "gpu": "Google Colab T4 (remote runtime, ~14.5-14.6GB peak observed for HGT/HAN)",
    "ram_gb": null,
    "os": "Linux (Colab remote runtime); orchestrated from Windows 10 via VS Code Colab extension"
  },
  "data": {
    "kg_name": "PrimeKG",
    "disease_nodes": 17080,
    "drug_nodes": 7957,
    "total_nodes": 129375,
    "total_edges": 8100498,
    "source": "Chandak et al., Scientific Data 2023, DOI: 10.7910/DVN/IXA7BM"
  },
  "models": [
    {
      "name": "TxGNN-architecture reimplementation (PyG), this study",
      "type": "Relational GNN (SAGEConv) + metric learning-style decoder",
      "pretrained": false,
      "note": "NOT the original authors' checkpoint. TxGNN pip package fails to install on current Colab Python (distutils removed). This is a from-scratch reimplementation of the published two-phase pretrain/fine-tune architecture.",
      "split_results": [
        {
          "split": "random",
          "seed": 42,
          "indication_AUPRC": 0.9911400736950633,
          "indication_AUROC": 0.9934462678079324,
          "contraindication_AUPRC": 0.9933417011464862,
          "contraindication_AUROC": 0.9955717390224496,
          "training_time_min": 1.15,
          "param_count": 9105600,
          "gpu_mem_peak_mb": null
        },
        {
          "split": "zero_shot",
          "seed": 42,
          "indication_AUPRC": 0.9799452567291346,
          "indication_AUROC": 0.9859299000351579,
          "contraindication_AUPRC": 0.9831985991764324,
          "contraindication_AUROC": 0.9892129765173643,
          "training_time_min": 1.16,
          "param_count": 9105600,
          "gpu_mem_peak_mb": null
        }
      ]
    },
    {
      "name": "HGT",
      "type": "Heterogeneous Graph Transformer (attention-based)",
      "pretrained": false,
      "note": "Single-stage supervised, reduced scale to fit a single T4 (hidden=32, heads=2, edges capped at 50k/relation, fp16). Not directly comparable to a full-scale run.",
      "split_results": [
        {
          "split": "random",
          "seed": 42,
          "indication_AUPRC": 0.9833719351637628,
          "indication_AUROC": 0.9859309588919655,
          "contraindication_AUPRC": 0.9944982619187841,
          "contraindication_AUROC": 0.9957607523210908,
          "training_time_min": 1.54,
          "param_count": 4327100,
          "gpu_mem_peak_mb": 14644.0
        },
        {
          "split": "zero_shot",
          "seed": 42,
          "indication_AUPRC": 0.9768662655766696,
          "indication_AUROC": 0.9829931501982594,
          "contraindication_AUPRC": 0.9935181390268848,
          "contraindication_AUROC": 0.9947407862116066,
          "training_time_min": 1.49,
          "param_count": 4327100,
          "gpu_mem_peak_mb": 14643.3
        }
      ]
    },
    {
      "name": "HAN",
      "type": "Heterogeneous Attention Network",
      "pretrained": false,
      "note": "Same reduced-scale setup as HGT (hidden=32, heads=2, edges capped at 50k/relation, fp16).",
      "split_results": [
        {
          "split": "random",
          "seed": 42,
          "indication_AUPRC": 0.9818071556711314,
          "indication_AUROC": 0.9860871080718548,
          "contraindication_AUPRC": 0.9812342987199284,
          "contraindication_AUROC": 0.9866273209255995,
          "training_time_min": 0.65,
          "param_count": 4169696,
          "gpu_mem_peak_mb": 14540.1
        },
        {
          "split": "zero_shot",
          "seed": 42,
          "indication_AUPRC": 0.9853306871105153,
          "indication_AUROC": 0.9882659195833461,
          "contraindication_AUPRC": 0.9855060222981411,
          "contraindication_AUROC": 0.9903145439124498,
          "training_time_min": 0.64,
          "param_count": 4169696,
          "gpu_mem_peak_mb": 14540.1
        }
      ]
    }
  ],
  "notes": "All metrics use random-negative-sampling evaluation, which is known to inflate AUPRC/AUROC because random non-edges are easy negatives. All three models land in a narrow 0.98-0.99 band on both metrics and both splits, which is itself a finding: this evaluation protocol does not discriminate well between architectures here. TxGNN ran at full PrimeKG scale (8.1M edges, hidden=64); HGT/HAN ran at reduced scale (edges capped at 50k/relation, hidden=32) after the first attempt OOM'd a T4 GPU \u2014 the hidden-dim and edge-coverage mismatch breaks strict apples-to-apples comparison and is flagged as a limitation, not glossed over."
}
```
