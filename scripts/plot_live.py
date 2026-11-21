"""Monitor live per simulazione_lume.py (lanciato automaticamente dalla scena).

Uso: python plot_live.py <porta> <csv_riferimento_moto_libero> <json_parametri>
Pannelli: DeltaY-u (lume vs moto libero), T-u, profilo forza di contatto lungo l'asse,
riepilogo con isteresi H.
"""
import csv
import json
import os
import socket
import sys
import threading
import time

import matplotlib.pyplot as plt
import numpy as np

PHASE_STYLE = {'loading': ('tab:blue', 'carico'), 'holding': ('tab:gray', 'mantenimento'),
               'unloading': ('tab:orange', 'scarico'), 'complete': ('tab:orange', None)}


class Receiver(threading.Thread):
    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port
        self.samples = []
        self.lock = threading.Lock()
        self.connected = False

    def run(self):
        sock = None
        for _ in range(600):
            try:
                sock = socket.create_connection(('127.0.0.1', self.port), timeout=2.0)
                break
            except OSError:
                time.sleep(0.2)
        if sock is None:
            return
        self.connected = True
        sock.settimeout(None)
        buffer = b''
        while True:
            try:
                chunk = sock.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
            *lines, buffer = buffer.split(b'\n')
            parsed = []
            for line in lines:
                try:
                    parsed.append(json.loads(line))
                except ValueError:
                    pass
            with self.lock:
                self.samples.extend(parsed)
        self.connected = False

    def snapshot(self):
        with self.lock:
            return list(self.samples)


def load_reference(path):
    if not os.path.isfile(path):
        return None
    with open(path, newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    return [(r['phase'], float(r['u']), float(r['delta_y']), float(r['tension']))
            for r in rows]


def plot_branches(ax, samples, key_x, key_y, labels=True, **kwargs):
    pts_all = [s for s in samples if key_x in s and key_y in s]
    known = [s for s in pts_all if s.get('phase') in PHASE_STYLE]
    if pts_all and not known:  # fasi non standard (es. simulazione_trave.py)
        ax.plot([s[key_x] for s in pts_all], [s[key_y] for s in pts_all],
                color='tab:blue', **kwargs)
        return
    for phase, (color, label) in PHASE_STYLE.items():
        pts = [(s[key_x], s[key_y]) for s in known if s['phase'] == phase]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, label=label if labels else None, **kwargs)


def hysteresis_gap(samples):
    """Scarto DeltaY_scarico(u) - DeltaY_carico(u) sul tratto gia' scaricato."""
    load = sorted((s['u'], s['delta_y']) for s in samples
                  if s.get('phase') == 'loading' and 'u' in s and 'delta_y' in s)
    unload = sorted((s['u'], s['delta_y']) for s in samples
                    if s.get('phase') in ('unloading', 'complete') and 'u' in s and 'delta_y' in s)
    if len(load) < 3 or len(unload) < 3:
        return None
    lu, ly = np.array(load).T
    uu, uy = np.array(unload).T
    grid = np.linspace(uu.min(), min(uu.max(), lu.max()), 200)
    return grid, np.interp(grid, uu, uy) - np.interp(grid, lu, ly)


def main():
    port = int(sys.argv[1])
    reference = load_reference(sys.argv[2]) if len(sys.argv) > 2 else None
    info = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    receiver = Receiver(port)
    receiver.start()

    title = ('Moto libero' if info.get('free_motion') else
             f"Lume D={info.get('diameter')} mm, mu={info.get('friction')}") + \
        f", k={info.get('spring_k')} N/mm"
    plt.ion()
    fig, ax_map = plt.subplot_mosaic([['dy', 'h', 't'], ['fu', 'ft', 'c'],
                                      ['hist', 'txt', 'txt']], figsize=(17, 12))
    fig.suptitle(title)
    ax_dy, ax_h, ax_t = ax_map['dy'], ax_map['h'], ax_map['t']
    ax_fu, ax_ft, ax_c = ax_map['fu'], ax_map['ft'], ax_map['c']
    ax_hist, ax_txt = ax_map['hist'], ax_map['txt']
    axes = np.array(list(ax_map.values()))
    ref_gap = hysteresis_gap([{'phase': p, 'u': u, 'delta_y': dy}
                              for p, u, dy, _ in reference]) if reference else None
    t_max = info.get('t_max', 10.0)

    while plt.fignum_exists(fig.number):
        samples = receiver.snapshot()
        for ax in axes.ravel():
            ax.clear()

        if reference and not info.get('free_motion'):
            for phase, style in (('loading', '--'), ('unloading', ':')):
                pts = [(u, dy) for p, u, dy, _ in reference if p == phase]
                if pts:
                    ax_dy.plot(*zip(*pts), color='k', ls=style, lw=1,
                               label=f'libero ({"carico" if phase == "loading" else "scarico"})')
                pts = [(u, t) for p, u, _, t in reference if p == phase]
                if pts:
                    ax_t.plot(*zip(*pts), color='k', ls=style, lw=1,
                          label='libero' if phase == 'loading' else None)

        if samples:
            plot_branches(ax_dy, samples, 'u', 'delta_y')
            plot_branches(ax_t, samples, 'u', 'tension')
            plot_branches(ax_t, samples, 'u', 'required_tension', labels=False, lw=0.8, alpha=0.4)
            last = samples[-1]
            profile = last.get('contact') or []
            if profile:
                cols = list(zip(*profile))  # x, |F|, F_normale, F_attrito
                ax_c.plot(cols[0], cols[1], 'o-', color='tab:red', label='|F| attuale')
                if len(cols) >= 4:
                    ax_c.plot(cols[0], cols[2], 's--', ms=3, color='tab:purple',
                              label='normale (radiale)')
                    ax_c.plot(cols[0], cols[3], '^:', ms=3, color='tab:brown',
                              label='attrito (assiale)')
            peak = max(samples, key=lambda s: s.get('contact_total', 0.0))
            if peak.get('contact') and peak.get('contact_total', 0.0) > 0.0:
                cols = list(zip(*peak['contact']))
                ax_c.plot(cols[0], cols[1], color='k', lw=1.5, ls='--',
                          label=f"|F| al picco (u={peak.get('u', 0.0):.2f} mm)")
            # Forza di contatto su TUTTO il ciclo: in funzione di u e del tempo
            for phase, (color, label) in PHASE_STYLE.items():
                ph = [s for s in samples if s.get('phase') == phase]
                if not ph:
                    continue
                us = [s.get('u', 0.0) for s in ph]
                ax_fu.plot(us, [s.get('contact_total', 0.0) for s in ph], color=color,
                           lw=2, label=f'|F| totale ({label})' if label else None)
                ax_fu.plot(us, [s.get('contact_normal', 0.0) for s in ph], color=color,
                           lw=1, ls='--')
                ax_fu.plot(us, [s.get('contact_friction', 0.0) for s in ph], color=color,
                           lw=1, ls=':')
            ax_fu.plot([], [], color='0.4', ls='--', lw=1, label='normale')
            ax_fu.plot([], [], color='0.4', ls=':', lw=1, label='attrito')
            ax_fu.plot(last.get('u', 0.0), last.get('contact_total', 0.0), 'o',
                       color='tab:red', ms=7)
            ts = [s.get('time', 0.0) for s in samples]
            ax_ft.plot(ts, [s.get('contact_total', 0.0) for s in samples], color='tab:red',
                       lw=2, label='|F| totale')
            ax_ft.plot(ts, [s.get('contact_normal', 0.0) for s in samples],
                       color='tab:purple', lw=1, ls='--', label='normale')
            ax_ft.plot(ts, [s.get('contact_friction', 0.0) for s in samples],
                       color='tab:brown', lw=1, ls=':', label='attrito')
            start = 0
            for i in range(1, len(samples) + 1):  # fondo colorato per fase
                if i == len(samples) or samples[i].get('phase') != samples[start].get('phase'):
                    color = PHASE_STYLE.get(samples[start].get('phase'), ('0.9', None))[0]
                    ax_ft.axvspan(ts[start], ts[i - 1], color=color, alpha=0.08, lw=0)
                    start = i
            # Valore corrente dell'isteresi H nel pannello DeltaY-u
            h_last = last.get('hysteresis', 0.0)
            ax_dy.text(0.02, 0.95, f'H = {h_last:.4f} mm', transform=ax_dy.transAxes,
                       va='top', fontsize=9,
                       bbox=dict(boxstyle='round', fc='white', alpha=0.8))
            # Pannello isteresi: scarto scarico-carico in funzione di u
            gap = hysteresis_gap(samples)
            if ref_gap is not None and not info.get('free_motion'):
                ax_h.plot(*ref_gap, color='k', ls='--', lw=1,
                          label=f'libero: H={np.abs(ref_gap[1]).max():.4f} mm')
            if gap is not None:
                ax_h.plot(*gap, color='tab:orange', lw=2,
                          label=f"attuale: H={last.get('hysteresis', 0.0):.4f} mm")
                ax_h.fill_between(gap[0], 0, gap[1], color='tab:orange', alpha=0.2)
                i_max = int(np.argmax(np.abs(gap[1])))
                ax_h.plot(gap[0][i_max], gap[1][i_max], 'o', color='tab:red', ms=7)
            else:
                ax_h.text(0.5, 0.5, 'disponibile durante lo scarico', ha='center',
                          transform=ax_h.transAxes, color='0.4')
            # Andamento di H nel tempo
            hs = [(s.get('time', 0.0), s.get('hysteresis', 0.0)) for s in samples]
            ax_hist.plot(*zip(*hs), color='tab:green')
            sat = next((s['u'] for s in samples if s.get('saturated')), None)
            ax_txt.axis('off')
            ax_txt.text(0.02, 0.95, '\n'.join([
                f"t = {last.get('time', 0.0):.2f} s   fase: {last.get('phase', '-')}",
                f"u = {last.get('u', 0.0):.3f} mm",
                f"DeltaY punta = {last.get('delta_y', 0.0):.4f} mm",
                f"T applicata = {last.get('tension', 0.0):.3f} N   "
                f"(lambda cavo = {last.get('cable_lambda', float('nan')):.3f} N)",
                f"T richiesta k(u-dc) = {last.get('required_tension', float('nan')):.3f} N",
                f"Forza contatto totale = {last.get('contact_total', 0.0):.4f} N",
                f"  normale = {last.get('contact_normal', 0.0):.4f} N",
                f"  attrito = {last.get('contact_friction', 0.0):.4f} N",
                f"  sezioni a contatto = {last.get('contact_sections', 0)}",
                '',
                f"Isteresi H = {last.get('hysteresis', 0.0):.4f} mm",
                f"Area ciclo = {last.get('hysteresis_area', float('nan')):.4f} mm^2",
                '',
                (f"SATURAZIONE: T_richiesta > T_max={t_max:g} N da u = {sat:.2f} mm"
                 if sat is not None else f"T_max = {t_max:g} N non raggiunta"),
            ]), va='top', family='monospace', fontsize=10, transform=ax_txt.transAxes)
        else:
            ax_txt.axis('off')
            ax_txt.text(0.05, 0.9, 'In attesa dei dati da SOFA (premi Animate)...',
                        transform=ax_txt.transAxes)

        ax_dy.set(title='DeltaY punta - u', xlabel='u motore [mm]', ylabel='DeltaY [mm]')
        ax_t.axhline(t_max, color='r', lw=0.8, ls='--')
        ax_t.set(title='Tensione del tendine - u', xlabel='u motore [mm]', ylabel='T [N]')
        ax_c.set(title="Forza di contatto lungo l'asse (per sezione)",
                 xlabel='x [mm]', ylabel='F contatto per sezione [N]', xlim=(0, 100))
        ax_c.axvspan(0, 20, color='0.85', zorder=0)  # tratto incastrato
        ax_h.axhline(0, color='k', lw=0.6)
        ax_h.set(title='Isteresi: ΔY_scarico − ΔY_carico',
                 xlabel='u motore [mm]', ylabel='scarto [mm]  (H = max |scarto|)')
        ax_hist.set(title='Isteresi H nel tempo', xlabel='t [s]', ylabel='H [mm]')
        ax_fu.set(title='Forza di contatto totale - u (tutto il ciclo)',
                  xlabel='u motore [mm]', ylabel='F contatto [N]')
        ax_ft.set(title='Forza di contatto nel tempo (fondo: blu carico, arancio scarico)',
                  xlabel='t [s]', ylabel='F contatto [N]')
        for ax in (ax_dy, ax_h, ax_t, ax_c, ax_hist, ax_fu, ax_ft):
            ax.grid(True)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        fig.tight_layout()
        snapshot = os.environ.get('PLOT_LIVE_SNAPSHOT')  # solo per test: salva ed esce
        if snapshot and samples and not receiver.connected:
            fig.savefig(snapshot, dpi=90)
            break
        plt.pause(0.3)


if __name__ == '__main__':
    main()
