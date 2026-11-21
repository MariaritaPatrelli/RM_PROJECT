"""Esegue tutte le combinazioni del Progetto 16 in batch (senza GUI).

Uso: python run_batch.py [percorso\\runSofa.exe] [--jobs N]
Prima i casi in moto libero (uno per k), poi D x mu x k nel lume.
Risultati in ../data/run_*.csv e ../data/summary_*.json (log in ../data/logs);
poi lanciare plot_summary.py.
"""
import argparse
import itertools
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
SCENE = os.path.join(PROJECT, 'simulazione_lume.py')
LOG_DIR = os.path.join(PROJECT, 'data', 'logs')
DEFAULT_RUNSOFA = os.path.join(os.path.expanduser('~'), 'Desktop', 'SOFA_v26.06.00_Win64',
                               'bin', 'runSofa.exe')
DIAMETERS = (3.0, 4.0, 6.0)
FRICTIONS = (0.1, 0.3)
SPRINGS = (0.6, 1.2, 4.8)
U_MAX = 10.0
FEED = 0.35
HOLD = 0.5
DT = 0.01


def tag_of(env_extra):
    k = float(env_extra['SOFA_SPRING_K'])
    if env_extra['SOFA_FREE_MOTION'] == '1':
        return f'free_k{k:g}'
    return (f"D{float(env_extra['SOFA_LUMEN_DIAMETER']):g}_"
            f"mu{float(env_extra['SOFA_FRICTION']):g}_k{k:g}")


def run(runsofa, env_extra):
    env = dict(os.environ, SOFA_NO_MONITOR='1', SOFA_TARGET_OFFSET=str(U_MAX),
               SOFA_FEED_RATE=str(FEED), SOFA_HOLD_TIME=str(HOLD), **env_extra)
    steps = math.ceil((2 * U_MAX / FEED + HOLD) / DT) + 100
    cmd = [runsofa, '-l', 'SofaPython3', '-g', 'batch', '-n', str(steps), SCENE]
    tag = tag_of(env_extra)
    start = time.time()
    print(f'>> avvio {tag}', flush=True)
    with open(os.path.join(LOG_DIR, f'{tag}.log'), 'w', encoding='utf-8',
              errors='replace') as log:
        subprocess.run(cmd, env=env, check=False, stdout=log, stderr=subprocess.STDOUT,
                       cwd=PROJECT)
    ok = os.path.isfile(os.path.join(PROJECT, 'data', f'summary_{tag}.json'))
    print(f'<< {tag}: {"OK" if ok else "ERRORE (vedi log)"} in {time.time() - start:.0f} s',
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runsofa', nargs='?', default=None)
    parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    runsofa = (args.runsofa or shutil.which('runSofa') or shutil.which('runSofa.exe')
               or (DEFAULT_RUNSOFA if os.path.isfile(DEFAULT_RUNSOFA) else None))
    if not runsofa:
        raise SystemExit('runSofa non trovato: passa il percorso come argomento.')
    os.makedirs(LOG_DIR, exist_ok=True)
    cases = [{'SOFA_FREE_MOTION': '1', 'SOFA_SPRING_K': str(k)} for k in SPRINGS]
    cases += [{'SOFA_FREE_MOTION': '0', 'SOFA_LUMEN_DIAMETER': str(d),
               'SOFA_FRICTION': str(mu), 'SOFA_SPRING_K': str(k)}
              for d, mu, k in itertools.product(DIAMETERS, FRICTIONS, SPRINGS)]
    print(f'{len(cases)} casi, {args.jobs} in parallelo', flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(lambda c: run(runsofa, c), cases))


if __name__ == '__main__':
    main()
