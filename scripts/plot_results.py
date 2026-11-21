"""Visualizzazione live dei risultati della scena SOFA."""
import csv
import os
import sys
import time

import matplotlib.pyplot as plt


def read_csv(path):
    if not os.path.isfile(path):
        return []
    try:
        with open(path, newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error, UnicodeDecodeError):
        return []


def to_float(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return None


def update_plot(ax_delta, ax_contact, ax_hysteresis, history_path, contact_path):
    history = read_csv(history_path)
    contacts = read_csv(contact_path)

    ax_delta.clear()
    if history:
        u = [to_float(row, 'u_mm') for row in history]
        delta_y = [to_float(row, 'DeltaY_punta_mm') for row in history]
        points = [(x, y) for x, y in zip(u, delta_y)
                  if x is not None and y is not None]
        if points:
            x_values, y_values = zip(*points)
            ax_delta.plot(x_values, y_values, color='tab:blue')
    ax_delta.set_title('Risposta del tendine')
    ax_delta.set_xlabel('u [mm]')
    ax_delta.set_ylabel('deltaY [mm]')
    ax_delta.grid(True)

    ax_contact.clear()
    if contacts:
        latest_time = to_float(contacts[-1], 'tempo_s')
        latest = [row for row in contacts
                  if to_float(row, 'tempo_s') == latest_time]
        x_values = [to_float(row, 'X_mm') for row in latest]
        force_values = [to_float(row, 'modulo_N') for row in latest]
        points = [(x, force) for x, force in zip(x_values, force_values)
                  if x is not None and force is not None]
        if points:
            points.sort()
            x_values, force_values = zip(*points)
            ax_contact.plot(x_values, force_values, color='tab:red')
    ax_contact.set_title('Profilo forza di contatto')
    ax_contact.set_xlabel('X [mm]')
    ax_contact.set_ylabel('|F_contatto| [N]')
    ax_contact.grid(True)

    ax_hysteresis.clear()
    if history:
        delta_ls = [to_float(row, 'DeltaLs_mm') for row in history]
        hysteresis = [to_float(row, 'H_N') for row in history]
        points = [(x, y) for x, y in zip(delta_ls, hysteresis)
                  if x is not None and y is not None]
        if points:
            x_values, h_values = zip(*points)
            ax_hysteresis.plot(x_values, h_values, color='tab:green')
    ax_hysteresis.set_title('Isteresi H')
    ax_hysteresis.set_xlabel('DeltaLs [mm]')
    ax_hysteresis.set_ylabel('H [N]')
    ax_hysteresis.grid(True)


def main():
    if len(sys.argv) != 3:
        raise SystemExit('Uso: plot_results.py dati_simulazione.csv distribuzione_contatto.csv')

    plt.ion()
    figure, axes = plt.subplots(3, 1, figsize=(9, 11))
    try:
        while plt.fignum_exists(figure.number):
            update_plot(axes[0], axes[1], axes[2], sys.argv[1], sys.argv[2])
            figure.tight_layout()
            figure.canvas.draw()
            figure.canvas.flush_events()
            time.sleep(0.5)
    finally:
        plt.close(figure)


if __name__ == '__main__':
    main()
