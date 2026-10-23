#!/usr/bin/env python3
"""Turns the CSV results in ../data into the LaTeX tables and pgfplots data of the paper."""
import csv
import math
import os

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, '..', 'data')
tables = os.path.join(here, '..', 'tables')
os.makedirs(tables, exist_ok=True)


def sci(x, digits=1):
    """3.4e-11 -> $3.4\\cdot 10^{-11}$; values >= 0.01 are printed plainly."""
    x = float(x)
    if x == 0:
        return '$0$'
    if x >= 0.01:
        return f'${x:.2f}$'
    e = math.floor(math.log10(x))
    m = x / 10 ** e
    if round(m, digits) >= 10:
        m, e = m / 10, e + 1
    return f'${m:.{digits}f}\\cdot 10^{{{e}}}$'


# Table: charts
rows = list(csv.DictReader(open(os.path.join(data, 'charts.csv'))))
names = {
    'identity': 'identity',
    'pi about (1,2,3)': r'$\pi$ about $(1,2,3)$',
    'pi/2 about z': r'$\pi/2$ about $e_3$',
    '2pi/3 about (1,1,0)': r'$2\pi/3$ about $(1,1,0)$',
    '1 rad about (1,2,3)': r'$1$ rad about $(1,2,3)$',
}
with open(os.path.join(tables, 'charts.tex'), 'w') as f:
    for r in rows:
        q = '(' + ', '.join(f'{float(r[k]):.2f}' for k in 'wxyz') + ')'
        cells = [sci(r[k]) for k in ('scalar', 'z_chart', 'alternating', 'best', 'eigensolver')]
        f.write(f"{names[r['rotation']]} & " + ' & '.join(cells) + ' \\\\\n')

# Table: accuracy
rows = list(csv.DictReader(open(os.path.join(data, 'accuracy.csv'))))
with open(os.path.join(tables, 'accuracy.tex'), 'w') as f:
    for r in rows:
        noise = float(r['noise'])
        truth = sci(r['max_angle_vs_truth']) if noise == 0 else '--'
        excess = sci(r['max_rel_excess_loss']) if noise > 0 else '--'
        label = '$0$' if noise == 0 else sci(noise, 0) if noise < 0.01 else f'${noise:g}$'
        f.write(f"{label} & {sci(r['max_angle_vs_eigensolver'])} & {excess} & {truth} & "
                f"${float(r['mean_newton_steps']):.2f}$ & ${r['max_newton_steps']}$ \\\\\n")

# Timing: table at selected N and wide CSV for the plot
order = ['Proposed', 'QUEST', 'FOAM', 'FLAE Newton', 'FLAE', 'Symbolic', 'FA3R', 'FA3R Int',
         'Q-Method', 'Horn', 'SVD', 'SVD McAdams', 'GA Valkenburg']
t = {}
for opt in ('O3', 'Ofast'):
    for r in csv.DictReader(open(os.path.join(data, f'timing_{opt}.csv'))):
        t[(opt, r['method'], int(r['n']))] = float(r['ns'])
sel = [3, 10, 100, 1000]
with open(os.path.join(tables, 'timing.tex'), 'w') as f:
    for m in order:
        cells = []
        for opt in ('O3', 'Ofast'):
            vals = [t[(opt, m, n)] for n in sel]
            best = [min(t[(opt, mm, n)] for mm in order) for n in sel]
            cells += [f'$\\mathbf{{{v:.0f}}}$' if v == b else f'${v:.0f}$' for v, b in zip(vals, best)]
        name = r'\textbf{Proposed}' if m == 'Proposed' else m
        f.write(f'{name} & ' + ' & '.join(cells) + ' \\\\\n')
ns = sorted({n for (_, _, n) in t})
for opt in ('O3', 'Ofast'):
    with open(os.path.join(data, f'timing_{opt}_wide.csv'), 'w') as f:
        f.write('n,' + ','.join(m.replace(' ', '').replace('-', '') for m in order) + '\n')
        for n in ns:
            f.write(f'{n},' + ','.join(f'{t[(opt, m, n)]:.1f}' for m in order) + '\n')
print('tables written to', os.path.normpath(tables))
