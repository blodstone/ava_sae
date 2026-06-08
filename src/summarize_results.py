import re
import sys

ALPHA = 0.000250

def parse_results(filepath):
    with open(filepath) as f:
        content = f.read()

    blocks = re.split(r'=== Linguistic term: ([\w-]+) ===', content)
    phenomena = {}
    for i in range(1, len(blocks), 2):
        term = blocks[i].strip()
        block = blocks[i+1]

        # Parse per-layer rows
        layers = []
        for line in block.splitlines():
            m = re.match(
                r'\s+(\S+)\s+'
                r'([\d.eE+\-]+)\s+'   # avg_pref_cv
                r'([\d.eE+\-]+)\s+'   # avg_pref_wf
                r'([\d.eE+\-]+)\s+'   # avg_ctrl_cv
                r'([\d.eE+\-]+)\s+'   # avg_ctrl_wf
                r'([\d.eE+\-]+)\s+'   # avg_sel_cv
                r'([\d.eE+\-]+)\s+'   # avg_sel_wf
                r'([\d.eE+\-]+)\s+'   # Wcv_p
                r'([\d.eE+\-]+)\s+'   # Www_p
                r'([\d.eE+\-]+)\s+'   # Wsel_p
                r'([\d.eE+\-]+)\s+'   # delta_acc_abl
                r'([\d.eE+\-]+)\s+'   # delta_acc_rnd
                r'(\w+)\s+'           # cv>0
                r'(\w+)\s+'           # wf~0
                r'(\w+)',             # sel>0
                line
            )
            if m:
                layers.append({
                    'layer':        m.group(1),
                    'avg_pref_cv':  float(m.group(2)),
                    'avg_pref_wf':  float(m.group(3)),
                    'avg_ctrl_cv':  float(m.group(4)),
                    'avg_ctrl_wf':  float(m.group(5)),
                    'avg_sel_cv':   float(m.group(6)),
                    'avg_sel_wf':   float(m.group(7)),
                    'Wcv_p':        float(m.group(8)),
                    'Www_p':        float(m.group(9)),
                    'Wsel_p':       float(m.group(10)),
                    'delta_acc_abl':float(m.group(11)),
                    'delta_acc_rnd':float(m.group(12)),
                    'cv_gt0':       m.group(13) == 'YES',
                    'wf_near0':     m.group(14) == 'YES',
                    'sel_gt0':      m.group(15) == 'YES',
                })
        phenomena[term] = layers
    return phenomena


def row_criteria(layer):
    """
    Returns (passes, significant) for each of F1, F2, F3 for a single row.
    passes:      the criterion holds
    significant: the corresponding Wilcoxon p-value < ALPHA
    """
    f1 = layer['cv_gt0'],       layer['Wcv_p'] < ALPHA
    f2 = abs(layer['avg_pref_wf']) < 0.05, layer['Www_p'] < ALPHA
    f3 = layer['sel_gt0'],      layer['Wsel_p'] < ALPHA
    return f1, f2, f3  # each is (passes, sig)


def compute_criteria(phenomena):
    """
    For each phenomenon and each criterion combination, check across all layers.
    A phenomenon:
      - 'passes'  if any row satisfies all criteria in the combination
      - 'sig'     if any row satisfies all criteria AND all corresponding p-values < ALPHA
    """
    results = {}
    for term, layers in phenomena.items():
        rows = [row_criteria(l) for l in layers]
        # rows[i] = ((f1_passes, f1_sig), (f2_passes, f2_sig), (f3_passes, f3_sig))

        def check(combos):
            """combos: list of (criterion_index) to AND together. 0=F1,1=F2,2=F3"""
            passes = any(all(rows[i][c][0] for c in combos) for i in range(len(rows)))
            sig    = any(all(rows[i][c][0] and rows[i][c][1] for c in combos) for i in range(len(rows)))
            return passes, sig

        results[term] = {
            'F1':      check([0]),
            'F2':      check([1]),
            'F3':      check([2]),
            'F1F2':    check([0, 1]),
            'F2F3':    check([1, 2]),
            'F1F3':    check([0, 2]),
            'F1F2F3':  check([0, 1, 2]),
        }
    return results


def count_passing(results, key):
    """Returns (n_significant, n_total) across all phenomena for a given key."""
    total = sum(1 for t in results if results[t][key][0])
    sig   = sum(1 for t in results if results[t][key][1])
    return sig, total


def fmt(sig, total):
    return f'{sig}/{total}' if total > 0 else '0/0'


def make_latex_table(col_name, counts):
    rows = [
        (r'$F_1$',         'F1'),
        (r'$F_2$',         'F2'),
        (r'$F_3$',         'F3'),
        (r'$F_1 F_2$',     'F1F2'),
        (r'$F_2 F_3$',     'F2F3'),
        (r'$F_1 F_3$',     'F1F3'),
        (r'$F_1 F_2 F_3$', 'F1F2F3'),
    ]
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\small',
        r'\setlength{\tabcolsep}{6pt}',
        r'\caption{Number of phenomena (out of 13) passing each falsification criterion. '
        r'$\alpha = 0.001/|\mathcal{L}|$ (Bonferroni). Top-$k = 15$.}',
        r'\label{tab:summary}',
        r'\begin{tabular}{lc}',
        r'\toprule',
        f'& {col_name} \\\\',
        r'\midrule',
    ]
    for label, key in rows:
        lines.append(f'{label} & {fmt(*counts[key])} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'results.txt'
    col_name = sys.argv[2] if len(sys.argv) > 2 else 'Model'

    phenomena = parse_results(filepath)
    print(f"Parsed {len(phenomena)} phenomena: {list(phenomena.keys())}\n")

    results = compute_criteria(phenomena)

    print(f"{'Phenomenon':<30} {'F1':>7} {'F2':>7} {'F3':>7} {'F1F2':>7} {'F2F3':>7} {'F1F3':>7} {'F1F2F3':>9}")
    print("-" * 90)
    for term, vals in results.items():
        def cell(passes, sig):
            if not passes: return 'F'
            return 'T*' if sig else 'T'
        print(f"{term:<30} " + "  ".join(f"{cell(*v):>7}" for v in vals.values()))

    counts = {k: count_passing(results, k)
              for k in ['F1','F2','F3','F1F2','F2F3','F1F3','F1F2F3']}

    print("\nCounts (sig/total):")
    for k, (s, t) in counts.items():
        print(f"  {k:10s}: {s}/{t}")

    latex = make_latex_table(col_name, counts)
    print("\nLaTeX:\n")
    print(latex)

    with open('table_output.tex', 'w') as f:
        f.write(latex)
    print("\nSaved to table_output.tex")


if __name__ == '__main__':
    main()