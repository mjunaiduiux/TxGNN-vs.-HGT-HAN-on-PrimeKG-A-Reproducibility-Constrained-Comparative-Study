"""
Phase 4 — aggregate individual run JSONs into results.json and comparison_table.json.

Reads every per-run result file produced by Phases 1-3 and overwrites
results.json / comparison_table.json. Run this after every new experiment
so the aggregate files stay current. No values are invented here — anything
missing stays null.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def split_result(run, split_label):
    if run is None:
        return {
            'split': split_label,
            'seed': None,
            'indication_AUPRC': None,
            'indication_AUROC': None,
            'contraindication_AUPRC': None,
            'contraindication_AUROC': None,
            'training_time_min': None,
            'param_count': None,
        }
    return {
        'split': split_label,
        'seed': run.get('seed'),
        'indication_AUPRC': run.get('indication_AUPRC'),
        'indication_AUROC': run.get('indication_AUROC'),
        'contraindication_AUPRC': run.get('contraindication_AUPRC'),
        'contraindication_AUROC': run.get('contraindication_AUROC'),
        'training_time_min': run.get('training_time_min'),
        'param_count': run.get('param_count'),
        'gpu_mem_peak_mb': run.get('gpu_mem_peak_mb'),
    }


def main():
    data_card = load('data_card.json') or {}

    txgnn_random = load('txgnn_pyg_random.json')
    txgnn_zeroshot = load('txgnn_pyg_zero_shot.json')
    hgt_random = load('hgt_random_split.json')
    hgt_zeroshot = load('hgt_zeroshot_split.json')
    han_random = load('han_random_split.json')
    han_zeroshot = load('han_zeroshot_split.json')

    results = {
        'last_updated': '2026-06-29',
        'hardware': {
            'gpu': 'Google Colab T4 (remote runtime, ~14.5-14.6GB peak observed for HGT/HAN)',
            'ram_gb': None,
            'os': 'Linux (Colab remote runtime); orchestrated from Windows 10 via VS Code Colab extension',
        },
        'data': {
            'kg_name': data_card.get('kg_name', 'PrimeKG'),
            'disease_nodes': data_card.get('disease_nodes'),
            'drug_nodes': data_card.get('drug_nodes'),
            'total_nodes': data_card.get('total_nodes'),
            'total_edges': data_card.get('total_edges'),
            'source': data_card.get('source'),
        },
        'models': [
            {
                'name': 'TxGNN-architecture reimplementation (PyG), this study',
                'type': 'Relational GNN (SAGEConv) + metric learning-style decoder',
                'pretrained': False,
                'note': 'NOT the original authors\' checkpoint. TxGNN pip package fails to install on current Colab Python (distutils removed). This is a from-scratch reimplementation of the published two-phase pretrain/fine-tune architecture.',
                'split_results': [
                    split_result(txgnn_random, 'random'),
                    split_result(txgnn_zeroshot, 'zero_shot'),
                ],
            },
            {
                'name': 'HGT',
                'type': 'Heterogeneous Graph Transformer (attention-based)',
                'pretrained': False,
                'note': 'Single-stage supervised, reduced scale to fit a single T4 (hidden=32, heads=2, edges capped at 50k/relation, fp16). Not directly comparable to a full-scale run.',
                'split_results': [
                    split_result(hgt_random, 'random'),
                    split_result(hgt_zeroshot, 'zero_shot'),
                ],
            },
            {
                'name': 'HAN',
                'type': 'Heterogeneous Attention Network',
                'pretrained': False,
                'note': 'Same reduced-scale setup as HGT (hidden=32, heads=2, edges capped at 50k/relation, fp16).',
                'split_results': [
                    split_result(han_random, 'random'),
                    split_result(han_zeroshot, 'zero_shot'),
                ],
            },
        ],
        'notes': (
            'All metrics use random-negative-sampling evaluation, which is known to inflate AUPRC/AUROC '
            'because random non-edges are easy negatives. All three models land in a narrow 0.98-0.99 band '
            'on both metrics and both splits, which is itself a finding: this evaluation protocol does not '
            'discriminate well between architectures here. TxGNN ran at full PrimeKG scale (8.1M edges, '
            'hidden=64); HGT/HAN ran at reduced scale (edges capped at 50k/relation, hidden=32) after the '
            'first attempt OOM\'d a T4 GPU — the hidden-dim and edge-coverage mismatch breaks strict '
            'apples-to-apples comparison and is flagged as a limitation, not glossed over.'
        ),
    }

    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    def row(dim, hgt_han_val, txgnn_val):
        return [dim, hgt_han_val, txgnn_val]

    comparison_table = {
        'columns': ['Dimension', 'Generic GNN (HGT/HAN)', 'TxGNN'],
        'rows': [
            row('Task framing', 'Per-disease supervised link prediction',
                f"Unified indication+contraindication across {data_card.get('disease_nodes', 'N/A')} diseases"),
            row('Training paradigm', 'Single-stage supervised', 'Two-phase: self-supervised pretrain -> fine-tune'),
            row('Attention mechanism', 'Core (multi-head self-attention per node/metapath)', 'Not used (SAGEConv mean aggregation)'),
            row('Zero-shot capability', 'Not designed for it', 'Designed for it'),
            row('Random-split indication AUPRC (this study)',
                hgt_random.get('indication_AUPRC') if hgt_random else None,
                txgnn_random.get('indication_AUPRC') if txgnn_random else None),
            row('Zero-shot indication AUPRC (this study)',
                hgt_zeroshot.get('indication_AUPRC') if hgt_zeroshot else None,
                txgnn_zeroshot.get('indication_AUPRC') if txgnn_zeroshot else None),
            row('Parameter count (this study)',
                hgt_random.get('param_count') if hgt_random else None,
                txgnn_random.get('param_count') if txgnn_random else None),
            row('Training time, random split (min, this study)',
                hgt_random.get('training_time_min') if hgt_random else None,
                txgnn_random.get('training_time_min') if txgnn_random else None),
            row('Interpretability', 'Attention weights (limited multi-hop)', 'Multi-hop explainer paths (not evaluated in this study)'),
        ],
        'notes': (
            'Numeric rows are from this study\'s reduced-scale runs (HGT/HAN capped at 50k edges/relation, '
            'hidden=32; TxGNN reimplementation at full scale, hidden=64) — not directly comparable in scale, '
            'see results.json notes.'
        ),
    }

    with open(os.path.join(HERE, 'comparison_table.json'), 'w') as f:
        json.dump(comparison_table, f, indent=2)

    print('Wrote results.json and comparison_table.json')


if __name__ == '__main__':
    main()
