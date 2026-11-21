"""Grafici e tabella finale del Progetto 16 a partire da ../data/summary_*.json e run_*.csv.

Figure prodotte in ../data:
- fig_deltaY_u.png        DeltaY-u nel lume vs moto libero (griglia mu x D, una curva per k)
- fig_contact_profile.png profilo della forza di contatto lungo l'asse (|F|, normale, attrito)
- fig_isteresi.png        cicli di isteresi DeltaY-u: moto libero vs lume mu=0.1 / mu=0.3
- fig_H.png               H in funzione di D, mu e k + aumento di H rispetto al moto libero
- fig_tensione.png        tensione richiesta T-u nel lume vs moto libero, con T_max
- tabella_target.csv      k ottimo per (D, mu), aumento di H, saturazione a T_max
Uso: python plot_summary.py [--no-show]
"""
import csv
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
T_MAX = 10.0
RIGID_LENGTH = 20.0
# Palette categoriale validata (slot 1-3) per k; mu codificato con lo stile di linea.
K_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6']
MU_COLORS = {0.1: '#2a78d6', 0.3: '#eb6834'}
FREE_COLOR = '#52514e'
MU_STYLE = {0.1: '-', 0.3: '--'}
plt.rcParams.update({'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'lines.linewidth': 1.6, 'font.size': 9})


def load_run(tag):
    path = os.path.join(DATA, f'run_{tag}.csv')
    if not os.path.isfile(path):
        return []
    with open(path, newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def branch(rows, phases, key='delta_y'):
    pts = [(float(r['u']), float(r[key])) for r in rows if r['phase'] in phases]
    return zip(*pts) if pts else ([], [])


def plot_loop(ax, rows, color, label, lw=1.6, alpha=1.0):
    """Ciclo carico (continuo) / scarico (tratteggiato) con area del ciclo ombreggiata."""
    if not rows:
        return
    lu, ly = branch(rows, ('loading',))
    uu, uy = branch(rows, ('unloading', 'complete'))
    ax.plot(lu, ly, color=color, lw=lw, alpha=alpha, label=label)
    ax.plot(uu, uy, color=color, lw=lw, alpha=alpha, ls=(0, (4, 2)))
    lu, ly, uu, uy = map(np.asarray, (lu, ly, uu, uy))
    if len(lu) > 2 and len(uu) > 2:
        grid = np.linspace(0, min(lu.max(), uu.max()), 200)
        o_l, o_u = np.argsort(lu), np.argsort(uu)
        ax.fill_between(grid, np.interp(grid, lu[o_l], ly[o_l]),
                        np.interp(grid, uu[o_u], uy[o_u]), color=color, alpha=0.12 * alpha, lw=0)


def main():
    show = '--no-show' not in sys.argv
    summaries = []
    for path in glob.glob(os.path.join(DATA, 'summary_*.json')):
        with open(path, encoding='utf-8') as stream:
            summaries.append(json.load(stream))
    free = {s['spring_k']: s for s in summaries if s['free_motion']}
    lumen = [s for s in summaries if not s['free_motion']]
    if not lumen:
        raise SystemExit('Nessun risultato nel lume: eseguire prima run_batch.py')
    diameters = sorted({s['diameter'] for s in lumen})
    frictions = sorted({s['friction'] for s in lumen})
    springs = sorted({s['spring_k'] for s in lumen} | set(free))
    kcol = dict(zip(springs, K_COLORS))
    get = {(s['diameter'], s['friction'], s['spring_k']): s for s in lumen}
    runs = {key: load_run(s['tag']) for key, s in get.items()}
    free_runs = {k: load_run(f'free_k{k:g}') for k in springs}

    # 1) DeltaY-u: lume vs moto libero -------------------------------------
    fig, axes = plt.subplots(len(frictions), len(diameters), figsize=(4.6 * len(diameters),
                             3.8 * len(frictions)), squeeze=False, sharey=True, sharex=True)
    for i, mu in enumerate(frictions):
        for j, d in enumerate(diameters):
            ax = axes[i][j]
            for k in springs:
                plot_loop(ax, free_runs[k], FREE_COLOR, 'libero' if k == springs[0] else None,
                          lw=0.9, alpha=0.6)
                if (d, mu, k) in runs:
                    plot_loop(ax, runs[(d, mu, k)], kcol[k], f'lume k={k:g}')
            clear = next((s['clearance'] for s in lumen if s['diameter'] == d), None)
            if clear:
                ax.axhline(clear, color='#0b0b0b', lw=0.7, ls=':')
                ax.text(0.2, clear, ' gioco radiale', va='bottom', fontsize=7, color='#52514e')
            ax.set_title(f'D = {d:g} mm, μ = {mu:g}')
            if i == len(frictions) - 1:
                ax.set_xlabel('u motore [mm]')
            if j == 0:
                ax.set_ylabel('ΔY punta [mm]')
    axes[0][0].legend(fontsize=7, loc='upper left')
    fig.suptitle('ΔY – u: lume (colori) vs moto libero (grigio); '
                 'continuo = carico, tratteggio = scarico')
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, 'fig_deltaY_u.png'), dpi=150)

    # 2) Profilo della forza di contatto lungo l'asse (a u = u_max) ---------
    fig, axes = plt.subplots(2, len(diameters), figsize=(4.6 * len(diameters), 7.2),
                             squeeze=False, sharex=True)
    for j, d in enumerate(diameters):
        ax_f, ax_c = axes[0][j], axes[1][j]
        for mu in frictions:
            for k in springs:
                s = get.get((d, mu, k))
                prof = (s or {}).get('contact_profile_umax') or (s or {}).get('contact_profile_peak')
                if not prof:
                    continue
                cols = list(zip(*prof))
                ax_f.plot(cols[0], cols[1], color=kcol[k], ls=MU_STYLE.get(mu, '-'),
                          marker='o', ms=3.5, label=f'μ={mu:g}, k={k:g}')
                if len(cols) >= 4:
                    ax_c.plot(cols[0], cols[2], color=kcol[k], ls=MU_STYLE.get(mu, '-'),
                              marker='o', ms=3.5)
                    ax_c.plot(cols[0], cols[3], color=kcol[k], ls=MU_STYLE.get(mu, '-'),
                              marker='^', ms=3.5, alpha=0.55)
        for ax in (ax_f, ax_c):
            ax.axvspan(0, RIGID_LENGTH, color='#f0efec', zorder=0)
            ax.set_xlim(0, 101)
        ax_f.set_title(f'D = {d:g} mm  (u = u_max)')
        ax_c.set_xlabel('x lungo il tubo [mm]')
        if j == 0:
            ax_f.set_ylabel('|F contatto| per sezione [N]')
            ax_c.set_ylabel('normale ● / attrito ▲ [N]')
        if ax_f.get_legend_handles_labels()[0]:
            ax_f.legend(fontsize=7)
        else:
            ax_f.text(0.5, 0.5, 'nessun contatto', ha='center', transform=ax_f.transAxes)
    fig.suptitle("Distribuzione della forza di contatto lungo l'asse "
                 '(grigio = tratto incastrato; continuo μ=0.1, tratteggio μ=0.3)')
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, 'fig_contact_profile.png'), dpi=150)

    # 2b) Forza di contatto totale su tutto il ciclo (carico + scarico) --------
    fig, axes = plt.subplots(len(frictions), len(diameters), figsize=(4.6 * len(diameters),
                             3.8 * len(frictions)), squeeze=False, sharex=True)
    for i, mu in enumerate(frictions):
        for j, d in enumerate(diameters):
            ax = axes[i][j]
            for k in springs:
                rows = runs.get((d, mu, k))
                if not rows or 'contact_total' not in rows[0]:
                    continue
                lu, lf = branch(rows, ('loading', 'holding'), 'contact_total')
                uu, uf = branch(rows, ('unloading', 'complete'), 'contact_total')
                ax.plot(list(lu), list(lf), color=kcol[k], label=f'k={k:g}')
                ax.plot(list(uu), list(uf), color=kcol[k], ls=(0, (4, 2)))
                tu, tf = branch(rows, ('loading', 'holding', 'unloading', 'complete'),
                                'contact_friction')
                ax.plot(list(tu), list(tf), color=kcol[k], lw=0.8, ls=':', alpha=0.8)
            ax.set_title(f'D = {d:g} mm, μ = {mu:g}')
            if i == len(frictions) - 1:
                ax.set_xlabel('u motore [mm]')
            if j == 0:
                ax.set_ylabel('F contatto totale [N]')
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, loc='upper left')
            else:
                ax.text(0.5, 0.5, 'nessun dato', ha='center', transform=ax.transAxes)
    fig.suptitle('Forza di contatto durante tutto il ciclo: continuo = carico, '
                 'tratteggio = scarico, punteggiato = componente d\'attrito')
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, 'fig_forza_contatto_ciclo.png'), dpi=150)

    # 3) Cicli di isteresi: libero vs lume, per D (righe) e k (colonne) -----
    fig, axes = plt.subplots(len(diameters), len(springs), figsize=(4.4 * len(springs),
                             3.5 * len(diameters)), squeeze=False, sharex=True)
    for i, d in enumerate(diameters):
        for j, k in enumerate(springs):
            ax = axes[i][j]
            lines = []
            if k in free:
                plot_loop(ax, free_runs[k], FREE_COLOR, 'moto libero')
                lines.append(f"libero: H={free[k]['H']:.3f}")
            for mu in frictions:
                if (d, mu, k) in runs:
                    plot_loop(ax, runs[(d, mu, k)], MU_COLORS.get(mu, '#6250d6'),
                              f'lume μ={mu:g}')
                    s = get[(d, mu, k)]
                    lines.append(f"μ={mu:g}: H={s['H']:.3f}" +
                                 (f"  (sat. u={s['saturation_u']:.1f})" if s['saturated'] else ''))
            ax.text(0.98, 0.04, '\n'.join(lines) + '  [mm]', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=7, family='monospace',
                    bbox=dict(boxstyle='round', fc='white', ec='#d0cfca', alpha=0.9))
            ax.set_title(f'D = {d:g} mm, k = {k:g} N/mm')
            if i == len(diameters) - 1:
                ax.set_xlabel('u motore [mm]')
            if j == 0:
                ax.set_ylabel('ΔY punta [mm]')
    axes[0][0].legend(fontsize=7, loc='upper left')
    fig.suptitle('Isteresi ΔY–u (continuo = carico, tratteggio = scarico, '
                 'area = ciclo): effetto dell\'attrito e della molla')
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, 'fig_isteresi.png'), dpi=150)

    # 4) H vs D, mu, k e aumento rispetto al moto libero -------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    ax_d, ax_mu, ax_k, ax_dh = axes.ravel()
    for mu in frictions:
        for k in springs:
            pts = [(d, get[(d, mu, k)]['H']) for d in diameters if (d, mu, k) in get]
            if pts:
                ax_d.plot(*zip(*pts), marker='o', color=kcol[k], ls=MU_STYLE.get(mu, '-'),
                          label=f'μ={mu:g}, k={k:g}')
    for k in springs:
        if k in free:
            ax_d.axhline(free[k]['H'], color=kcol[k], lw=0.8, ls=':')
    for d, marker in zip(diameters, ('o', 's', 'D', 'v')):
        for k in springs:
            pts = [(mu, get[(d, mu, k)]['H']) for mu in frictions if (d, mu, k) in get]
            if pts:
                ax_mu.plot(*zip(*pts), marker=marker, color=kcol[k], label=f'D={d:g}, k={k:g}')
    for d, marker in zip(diameters, ('o', 's', 'D', 'v')):
        for mu in frictions:
            pts = [(k, get[(d, mu, k)]['H']) for k in springs if (d, mu, k) in get]
            if pts:
                ax_k.plot(*zip(*pts), marker=marker, ls=MU_STYLE.get(mu, '-'),
                          color=MU_COLORS.get(mu), label=f'D={d:g}, μ={mu:g}')
    if free:
        ks = sorted(free)
        ax_k.plot(ks, [free[k]['H'] for k in ks], color=FREE_COLOR, ls=':', marker='x',
                  lw=2, label='moto libero')
    # aumento di H rispetto al moto libero (stesso k), barre raggruppate
    groups = [(d, mu) for d in diameters for mu in frictions]
    width = 0.8 / max(1, len(springs))
    for n, k in enumerate(springs):
        vals = [get[(d, mu, k)]['H'] - free[k]['H'] if (d, mu, k) in get and k in free
                else np.nan for d, mu in groups]
        ax_dh.bar(np.arange(len(groups)) + (n - (len(springs) - 1) / 2) * width, vals,
                  width * 0.92, color=kcol[k], label=f'k={k:g}')
    ax_dh.axhline(0, color='#0b0b0b', lw=0.8)
    ax_dh.set_xticks(range(len(groups)), [f'D={d:g}\nμ={mu:g}' for d, mu in groups])
    ax_d.set(title='H vs diametro del lume (punteggiato = libero)', xlabel='D [mm]',
             ylabel='H [mm]')
    ax_mu.set(title='H vs coefficiente di attrito', xlabel='μ', ylabel='H [mm]')
    ax_k.set(title='H vs rigidezza della molla in serie', xlabel='k [N/mm]', ylabel='H [mm]')
    ax_dh.set(title='Aumento di H rispetto al moto libero (stesso k)', ylabel='ΔH [mm]')
    for ax in axes.ravel():
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, 'fig_H.png'), dpi=150)

    # 5) Tensione richiesta vs moto libero ---------------------------------
    fig, axes = plt.subplots(len(frictions), len(diameters), figsize=(4.6 * len(diameters),
                             3.6 * len(frictions)), squeeze=False, sharey=True, sharex=True)
    for i, mu in enumerate(frictions):
        for j, d in enumerate(diameters):
            ax = axes[i][j]
            for k in springs:
                xs, ys = branch(free_runs[k], ('loading',), 'required_tension')
                ax.plot(list(xs), list(ys), color=kcol[k], lw=0.9, ls=':')
                if (d, mu, k) in runs:
                    xs, ys = branch(runs[(d, mu, k)], ('loading',), 'required_tension')
                    ax.plot(list(xs), list(ys), color=kcol[k], label=f'k={k:g}')
            ax.axhline(T_MAX, color='#d03b3b', lw=0.9, ls='--')
            ax.text(0.2, T_MAX, ' T_max', va='bottom', fontsize=7, color='#d03b3b')
            ax.set_title(f'D = {d:g} mm, μ = {mu:g}')
            if i == len(frictions) - 1:
                ax.set_xlabel('u motore [mm]')
            if j == 0:
                ax.set_ylabel('T richiesta [N]')
    axes[0][0].legend(fontsize=7)
    fig.suptitle('Tensione richiesta k(u − ΔL) in carico: lume (continuo) '
                 'vs moto libero (punteggiato)')
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, 'fig_tensione.png'), dpi=150)

    # 6) Tabella target di progetto ----------------------------------------
    lines = ['D [mm];mu;k_ottimo [N/mm];H_min [mm];H_libero [mm];aumento H [mm];aumento H [%];'
             'F_contatto max [N];T_richiesta max [N];T_richiesta libero [N];'
             'casi saturati (k: u_sat)']
    print(f"{'D':>4} {'mu':>4} {'k*':>5} {'H_min':>8} {'H_lib':>8} {'dH':>8} {'dH%':>7} "
          f"{'Fc':>7} {'T_req':>6} {'T_lib':>6}  saturazione T_max")
    for d in diameters:
        for mu in frictions:
            cases = [get[(d, mu, k)] for k in springs if (d, mu, k) in get]
            if not cases:
                continue
            best = min(cases, key=lambda s: s['H'])
            fb = free.get(best['spring_k'], {})
            h_free = fb.get('H', float('nan'))
            dh = best['H'] - h_free
            pct = 100.0 * dh / h_free if h_free else float('nan')
            sat = ', '.join(f"k={s['spring_k']:g}: u={s['saturation_u']:.2f}"
                            for s in cases if s['saturated']) or 'nessuna'
            t_free = fb.get('required_tension_max', float('nan'))
            print(f"{d:4g} {mu:4g} {best['spring_k']:5g} {best['H']:8.4f} {h_free:8.4f} "
                  f"{dh:8.4f} {pct:7.1f} {best['contact_total_max']:7.3f} "
                  f"{best['required_tension_max']:6.2f} {t_free:6.2f}  {sat}")
            lines.append(f"{d:g};{mu:g};{best['spring_k']:g};{best['H']:.5f};{h_free:.5f};"
                         f"{dh:.5f};{pct:.1f};{best['contact_total_max']:.4f};"
                         f"{best['required_tension_max']:.3f};{t_free:.3f};{sat}")
    with open(os.path.join(DATA, 'tabella_target.csv'), 'w', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('Figure salvate in', DATA)
    if show:
        plt.show()


if __name__ == '__main__':
    main()
