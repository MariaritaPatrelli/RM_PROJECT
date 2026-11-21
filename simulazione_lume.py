# -*- coding: utf-8 -*-
"""Progetto 16 - Flessione in un lume: contatto lungo il corpo e attrito.

Catena: motore (corsa u) -> molla in serie k -> tendine (CableConstraint in forza)
-> tubo FEM continuo dentro un lume rigido con FrictionContactConstraint.

Avvio singolo (grafici live):
    runSofa -l SofaPython3 simulazione_lume.py
Parametri da variabili d'ambiente (default tra parentesi):
    SOFA_LUMEN_DIAMETER (3)   diametro interno del lume [mm]
    SOFA_FRICTION       (0.1) coefficiente di attrito mu
    SOFA_SPRING_K       (1.2) rigidezza molla in serie [N/mm]
    SOFA_FREE_MOTION    (0)   1 = moto libero (nessun lume)
    SOFA_TARGET_OFFSET  (10)  corsa massima del motore u_max [mm]
    SOFA_NO_MONITOR     (0)   1 = non aprire la finestra dei grafici (batch)
Campagna completa: python scripts/run_batch.py  (poi scripts/plot_summary.py)

Unita': mm, N, MPa, s.
"""
import csv
import json
import math
import os
import shutil
import socket
import subprocess
import sys

import numpy as np
import Sofa
import Sofa.Core

SOFA_ROOT = os.environ.get('SOFA_ROOT') or os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'SOFA_v26.06.00_Win64'))
for plugin_name in ('STLIB', 'SoftRobots'):
    plugin_python = os.path.join(SOFA_ROOT, 'plugins', plugin_name,
                                 'lib', 'python3', 'site-packages')
    if os.path.isdir(plugin_python) and plugin_python not in sys.path:
        sys.path.append(plugin_python)
from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.constraints import FixedBox


def environment_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError as exc:
        raise RuntimeError(f'{name} deve essere numerico.') from exc


# --- Geometria (coerente con mesh/tube_tetra.vtu: 16 sezioni x 9 nodi) -------
TUBE_LENGTH = 100.0
RIGID_LENGTH = 20.0                      # tratto incastrato (FixedBox)
FLEX_LENGTH = TUBE_LENGTH - RIGID_LENGTH  # tratto libero di flettere
OUTER_DIAMETER = 1.93
RIGID_TUBE_LENGTH = 100.0
TUBE_AXIAL_CELLS = 15
TUBE_NODES_PER_SECTION = 9

# --- Parametri di progetto ---------------------------------------------------
FREE_MOTION = os.environ.get('SOFA_FREE_MOTION', '0') == '1'
NO_MONITOR = os.environ.get('SOFA_NO_MONITOR', '0') == '1'
DEBUG = os.environ.get('SOFA_DEBUG', '0') == '1'
RIGID_TUBE_INNER_DIAMETER = environment_float('SOFA_LUMEN_DIAMETER', 3.0)
CONTACT_FRICTION = environment_float('SOFA_FRICTION', 0.1)
SPRING_K = environment_float('SOFA_SPRING_K', 1.2)
MAX_MOTOR_OFFSET = environment_float('SOFA_TARGET_OFFSET', 10.0)
MAX_TENDON_FORCE = 10.0                  # T_max [N]
E_LOAD_MPA = 77000.0
E_UNLOAD_MPA = 65000.0
POISSON_RATIO = 0.33
MATERIAL_SWITCH = os.environ.get('SOFA_MATERIAL_SWITCH', '1') == '1'

RIGID_TUBE_WALL_THICKNESS = 0.5
RIGID_TUBE_OUTER_DIAMETER = RIGID_TUBE_INNER_DIAMETER + 2.0 * RIGID_TUBE_WALL_THICKNESS
RIGID_TUBE_RADIAL_SEGMENTS = 32
CONTACT_ALARM_DISTANCE = 0.25  # [mm] distanza di rilevamento (prima della compenetrazione)
CONTACT_DISTANCE = 0.02        # [mm] distanza di contatto (gap residuo tubo-parete)
RIGID_TUBE_AXIAL_SEGMENTS = 100

TENDON_OFFSET = 0.6
CABLE_PULL_POINT = [RIGID_LENGTH, TENDON_OFFSET, 0.0]
SPRING_REST = 20.0
SPRING_DAMPING = 1.5
MOTOR_FEED_RATE = environment_float('SOFA_FEED_RATE', 0.35)  # mm/s (quasi-statico)
HOLD_TIME = environment_float('SOFA_HOLD_TIME', 0.5)
SIMULATION_DT = 0.01
RAYLEIGH_STIFFNESS = 0.05
RAYLEIGH_MASS = 0.08
MOTOR_SPHERE_RADIUS = 2.2
N_COILS = 9
N_SPRING_POINTS = 181
MOTOR_REST_X = -SPRING_REST

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, 'scripts')
MESH_DIR = os.path.join(PROJECT_DIR, 'mesh')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')


def run_tag():
    if FREE_MOTION:
        return f'free_k{SPRING_K:g}'
    return f'D{RIGID_TUBE_INNER_DIAMETER:g}_mu{CONTACT_FRICTION:g}_k{SPRING_K:g}'


def helix(p1, p2):
    """Punti della spirale (solo visuale) tra motore e ancoraggio."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    axis = p2 - p1
    if np.linalg.norm(axis) < 1e-9:
        axis = np.array([1.0, 0.0, 0.0])
    unit = axis / np.linalg.norm(axis)
    u = np.cross(unit, [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(unit, u)
    points = []
    taper = 0.045
    for i in range(N_SPRING_POINTS):
        s = i / (N_SPRING_POINTS - 1)
        angle = 2.0 * math.pi * N_COILS * s
        radius_factor = min(1.0, s / taper, (1.0 - s) / taper)
        point = p1 + s * axis + 2.5 * radius_factor * (
            math.cos(angle) * u + math.sin(angle) * v)
        points.append(point.tolist())
    points[0] = p1.tolist()
    points[-1] = p2.tolist()
    return points


def hysteresis_from_samples(samples):
    """H = max |DeltaY_carico(u) - DeltaY_scarico(u)| sul ciclo, e area del ciclo."""
    load = sorted((s['u'], s['delta_y']) for s in samples if s['phase'] == 'loading')
    unload = sorted((s['u'], s['delta_y']) for s in samples
                    if s['phase'] in ('unloading', 'complete'))
    if len(load) < 3 or len(unload) < 3:
        return 0.0, 0.0
    lu, ly = np.array(load).T
    uu, uy = np.array(unload).T
    # Solo il tratto di u gia' percorso in scarico: fuori da li' np.interp
    # estrapolerebbe a valore costante e darebbe un H fittizio durante lo scarico.
    u_lo, u_hi = max(lu.min(), uu.min()), min(lu.max(), uu.max())
    if u_hi <= u_lo:
        return 0.0, 0.0
    grid = np.linspace(u_lo, u_hi, 400)
    gap = np.interp(grid, uu, uy) - np.interp(grid, lu, ly)
    integrate = getattr(np, 'trapezoid', None) or np.trapz
    return float(np.max(np.abs(gap))), float(abs(integrate(gap, grid)))


class LumenController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraint_solver = kwargs['constraintSolver']
        self.motor = kwargs['motor']
        self.motor_visual = kwargs['motorVisual']
        self.anchor = kwargs['anchor']
        self.tip = kwargs['tip']
        self.spring_vis = kwargs['springVisual']
        self.tube_dofs = kwargs['tube'].dofs
        self.fem_ff = kwargs['tube'].forcefield
        self.cable = kwargs['cable']
        self.cable_dofs = kwargs['cableDofs']
        self.centerline_nodes = kwargs['centerlineNodes']
        self.spring_k = SPRING_K
        self.dt = SIMULATION_DT
        self.contact_warning_done = False
        self.monitor_client = None
        self.monitor_backlog = []
        self.monitor_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.monitor_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.monitor_server.bind(('127.0.0.1', 0))
        self.monitor_server.listen(1)
        self.monitor_server.setblocking(False)
        self.reset_cycle()

    # ------------------------------------------------------------------ stato
    def reset_cycle(self):
        self.offset = 0.0
        self.applied_tension = 0.0
        self.required_tension = 0.0
        self.cable_shortening = 0.0
        self.cable_length0 = None
        self.initial_tip_y = None
        self.simulation_time = 0.0
        self.phase = 'loading'
        self.phase_time = 0.0
        self.saturation_u = None
        self.samples = []
        self.max_profile = None
        self.max_profile_u = None
        self.umax_profile = None
        self.finalized = False
        self.set_young_modulus(E_LOAD_MPA)

    def set_young_modulus(self, value):
        current = float(np.ravel(self.fem_ff.youngModulus.value)[0])
        if abs(current - value) < 1e-9:
            return
        self.fem_ff.youngModulus.value = [value]
        # TetrahedronFEMForceField ricalcola le rigidezze solo con reinit().
        self.fem_ff.reinit()

    # ------------------------------------------------------------- grandezze
    def cable_length(self):
        pts = np.asarray(self.cable_dofs.position.value, dtype=float)
        path = np.vstack([CABLE_PULL_POINT, pts])
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    def centerline(self):
        pos = np.asarray(self.tube_dofs.position.value, dtype=float)
        last_node = self.centerline_nodes[-1][-1]
        if pos.ndim != 2 or len(pos) <= last_node:
            xs = np.linspace(0.0, TUBE_LENGTH, len(self.centerline_nodes))
            return np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
        return np.asarray([pos[nodes].mean(axis=0) for nodes in self.centerline_nodes])

    @staticmethod
    def constraint_triplets(data):
        """Matrice dei vincoli J come (righe, colonne, valori).

        In SOFA >= 24 `constraint.value` e' una scipy sparse (n_vincoli x 3*n_nodi,
        troncata all'ultima colonna non nulla). La stringa getValueString() invece
        e' su UNA riga con il numero di righe in testa: il vecchio parser la leggeva
        male e la forza di contatto risultava sempre nulla."""
        matrix = data.value
        if hasattr(matrix, 'tocoo'):
            coo = matrix.tocoo()
            return (np.asarray(coo.row, dtype=int), np.asarray(coo.col, dtype=int),
                    np.asarray(coo.data, dtype=float))
        # Fallback testuale: "<n_righe> {<riga> <n_blocchi> {<nodo> fx fy fz}*}*"
        tok = data.getValueString().split()
        rows, cols, vals = [], [], []
        pos, n_rows = 1, int(tok[0]) if tok else 0
        for _ in range(n_rows):
            row, n_blocks = int(tok[pos]), int(tok[pos + 1])
            pos += 2
            for _ in range(n_blocks):
                node = int(tok[pos])
                for c in range(3):
                    rows.append(row)
                    cols.append(3 * node + c)
                    vals.append(float(tok[pos + 1 + c]))
                pos += 4
        return np.array(rows, dtype=int), np.array(cols, dtype=int), np.array(vals, dtype=float)

    def read_constraint_forces(self):
        """Forze di contatto reali sui nodi FEM: F = J^T * lambda (righe del lume).

        Comprende la componente normale e quella d'attrito (righe tangenziali).
        Restituisce (forze_contatto[N,3], forza_cavo_lambda)."""
        n_nodes = len(self.tube_dofs.position.value)
        forces = np.zeros(3 * n_nodes)
        try:
            lam = np.ravel(np.asarray(self.constraint_solver.constraintForces.value, dtype=float))
            rows, cols, vals = self.constraint_triplets(self.tube_dofs.constraint)
            cable_rows = np.unique(self.constraint_triplets(self.cable_dofs.constraint)[0])
        except Exception as exc:  # noqa: BLE001 - diagnostica una sola volta
            if not self.contact_warning_done:
                print(f'[Lume] Impossibile leggere le forze di vincolo: {exc}')
                self.contact_warning_done = True
            return forces.reshape(-1, 3), float('nan')
        cable_lambda = float(sum(lam[r] for r in cable_rows if r < len(lam)))
        keep = (rows < len(lam)) & (cols < 3 * n_nodes) & ~np.isin(rows, cable_rows)
        np.add.at(forces, cols[keep], vals[keep] * lam[rows[keep]])
        return forces.reshape(-1, 3), cable_lambda

    # --------------------------------------------------------------- motore
    def move_motor(self):
        pose = [[MOTOR_REST_X - self.offset, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        self.motor.position.value = pose
        self.motor.velocity.value = [[0.0] * 6]
        self.motor_visual.position.value = pose
        self.update_spring_points()

    def update_spring_points(self):
        p1 = self.motor_visual.position.value[0][:3]
        p2 = self.anchor.position.value[0][:3]
        self.spring_vis.position.value = helix(p1, p2)

    def update_tip_marker(self):
        tip_pos = self.centerline()[-1]
        self.tip.position.value = [[float(tip_pos[0]), float(tip_pos[1]),
                                    float(tip_pos[2]), 0.0, 0.0, 0.0, 1.0]]

    # --------------------------------------------------------------- eventi
    def onAnimateBeginEvent(self, event):
        self.dt = float(event.get('dt', SIMULATION_DT))
        if self.cable_length0 is None:
            self.cable_length0 = self.cable_length()
        self.phase_time += self.dt
        if self.phase == 'loading':
            self.offset = min(MAX_MOTOR_OFFSET, self.offset + MOTOR_FEED_RATE * self.dt)
            if self.offset >= MAX_MOTOR_OFFSET:
                self.phase, self.phase_time = 'holding', 0.0
        elif self.phase == 'holding' and self.phase_time >= HOLD_TIME:
            self.phase, self.phase_time = 'unloading', 0.0
            if MATERIAL_SWITCH:
                self.set_young_modulus(E_UNLOAD_MPA)
        elif self.phase == 'unloading':
            self.offset = max(0.0, self.offset - MOTOR_FEED_RATE * self.dt)
            if self.offset <= 0.0:
                self.phase, self.phase_time = 'complete', 0.0

        # Molla in serie: T = k * (u - accorciamento del cavo), saturata a T_max.
        self.cable_shortening = self.cable_length0 - self.cable_length()
        self.required_tension = max(0.0, self.spring_k * (self.offset - self.cable_shortening))
        self.applied_tension = min(self.required_tension, MAX_TENDON_FORCE)
        if self.required_tension > MAX_TENDON_FORCE and self.saturation_u is None:
            self.saturation_u = self.offset
        self.cable.value.value = [self.applied_tension]
        self.move_motor()

    def onAnimateEndEvent(self, event):
        self.simulation_time += float(event.get('dt', self.dt))
        self.update_tip_marker()
        self.update_spring_points()
        if self.finalized:
            return

        # In moto libero non esistono righe di contatto: forces resta nullo.
        forces, cable_lambda = self.read_constraint_forces()
        line = self.centerline()
        # Forza risultante per sezione (9 nodi): modulo, componente normale
        # (radiale, piano y-z) e componente d'attrito (assiale, lungo x).
        section_vec = np.array([forces[nodes].sum(axis=0) for nodes in self.centerline_nodes])
        section_force = np.linalg.norm(section_vec, axis=1)
        section_normal = np.hypot(section_vec[:, 1], section_vec[:, 2])
        section_friction = np.abs(section_vec[:, 0])
        profile = [[float(x), float(f), float(fn), float(ft)] for x, f, fn, ft in
                   zip(line[:, 0], section_force, section_normal, section_friction)]
        if self.max_profile is None or section_force.sum() > sum(p[1] for p in self.max_profile):
            self.max_profile = profile
            self.max_profile_u = self.offset
        if self.phase == 'loading' or self.umax_profile is None:
            self.umax_profile = profile

        tip_y = float(line[-1, 1])
        if self.initial_tip_y is None:
            self.initial_tip_y = tip_y
        sample = {
            'time': self.simulation_time,
            'phase': self.phase,
            'u': self.offset,
            'delta_y': tip_y - self.initial_tip_y,
            'tension': self.applied_tension,
            'required_tension': self.required_tension,
            'cable_lambda': cable_lambda,
            'cable_shortening': self.cable_shortening,
            'spring_elongation': self.offset - self.cable_shortening,
            'saturated': self.required_tension > MAX_TENDON_FORCE,
            'contact_total': float(section_force.sum()),
            'contact_normal': float(section_normal.sum()),
            'contact_friction': float(section_friction.sum()),
            'contact_sections': int((section_force > 1e-6).sum()),
        }
        self.samples.append(sample)
        if DEBUG and len(self.samples) % 100 == 0:
            print(f"[Debug] t={sample['time']:.1f} u={sample['u']:.2f} T={sample['tension']:.2f} "
                  f"lam={cable_lambda:.2f} dY={sample['delta_y']:.4f} "
                  f"Fc={sample['contact_total']:.4f} tip={line[-1].round(3).tolist()}", flush=True)
        h_max, h_area = hysteresis_from_samples(self.samples)
        self.send_live_sample(dict(sample, hysteresis=h_max, hysteresis_area=h_area,
                                   contact=profile))
        if self.phase == 'complete':
            self.finalize(h_max, h_area)

    def onKeypressedEvent(self, event):
        if event.get('key', '') in ('0', 'z', 'Z'):
            self.reset_cycle()
            print('[Lume] ciclo azzerato')

    def onSimulationEndEvent(self, event):
        if self.monitor_client is not None:
            self.monitor_client.close()
        self.monitor_server.close()

    # ---------------------------------------------------------------- output
    def finalize(self, h_max, h_area):
        self.finalized = True
        os.makedirs(DATA_DIR, exist_ok=True)
        tag = run_tag()
        csv_path = os.path.join(DATA_DIR, f'run_{tag}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.samples[0].keys()))
            writer.writeheader()
            writer.writerows(self.samples)
        loading = [s for s in self.samples if s['phase'] == 'loading']
        summary = {
            'tag': tag,
            'free_motion': FREE_MOTION,
            'diameter': None if FREE_MOTION else RIGID_TUBE_INNER_DIAMETER,
            'friction': None if FREE_MOTION else CONTACT_FRICTION,
            'spring_k': self.spring_k,
            'u_max': MAX_MOTOR_OFFSET,
            'H': h_max,
            'H_area': h_area,
            'delta_y_max': max(s['delta_y'] for s in self.samples),
            'tension_max': max(s['tension'] for s in self.samples),
            'required_tension_max': max(s['required_tension'] for s in loading),
            'saturated': self.saturation_u is not None,
            'saturation_u': self.saturation_u,
            'contact_total_max': max(s['contact_total'] for s in self.samples),
            'contact_normal_max': max(s['contact_normal'] for s in self.samples),
            'contact_friction_max': max(s['contact_friction'] for s in self.samples),
            'contact_sections_max': max(s['contact_sections'] for s in self.samples),
            # profili: [x, |F|, F_normale, F_attrito] per sezione
            'contact_profile_peak': self.max_profile,
            'contact_profile_peak_u': self.max_profile_u,
            'contact_profile_umax': self.umax_profile,
            'clearance': None if FREE_MOTION else
            (RIGID_TUBE_INNER_DIAMETER - OUTER_DIAMETER) / 2.0 - CONTACT_DISTANCE,
        }
        with open(os.path.join(DATA_DIR, f'summary_{tag}.json'), 'w', encoding='utf-8') as stream:
            json.dump(summary, stream, indent=2)
        print(f'[Lume] ciclo completo {tag}: H={h_max:.4f} mm, area={h_area:.4f} mm^2, '
              f'T_max={summary["tension_max"]:.2f} N, saturato={summary["saturated"]} '
              f'-> {csv_path}')

    def send_live_sample(self, payload):
        line = (json.dumps(payload) + '\n').encode('utf-8')
        if self.monitor_client is None:
            if NO_MONITOR:
                return
            # Conserva i campioni finche' il monitor non si collega.
            self.monitor_backlog.append(line)
            try:
                self.monitor_client, _ = self.monitor_server.accept()
                self.monitor_client.setblocking(True)
            except BlockingIOError:
                return
            line = b''.join(self.monitor_backlog)
            self.monitor_backlog = []
        try:
            self.monitor_client.sendall(line)
        except OSError:
            self.monitor_client.close()
            self.monitor_client = None

    def launch_live_monitor(self):
        if NO_MONITOR:
            return
        python_exe = (os.environ.get('SOFA_MONITOR_PYTHON')
                      or shutil.which('python.exe') or shutil.which('pythonw.exe')
                      or shutil.which('python3'))
        if python_exe is None:
            print('[Lume] Interprete Python non trovato: grafici live disattivati.')
            return
        monitor_script = os.path.join(SCRIPTS_DIR, 'plot_live.py')
        reference = os.path.join(DATA_DIR, f'run_free_k{SPRING_K:g}.csv')
        port = self.monitor_server.getsockname()[1]
        info = json.dumps({'diameter': RIGID_TUBE_INNER_DIAMETER, 'friction': CONTACT_FRICTION,
                           'spring_k': SPRING_K, 'free_motion': FREE_MOTION,
                           't_max': MAX_TENDON_FORCE, 'u_max': MAX_MOTOR_OFFSET})
        os.makedirs(DATA_DIR, exist_ok=True)
        log_path = os.path.join(DATA_DIR, 'monitor_log.txt')
        # Errori del monitor in un file: lanciato da runSofa (GUI) non ha console.
        self.monitor_log = open(log_path, 'w', encoding='utf-8')
        self.monitor_process = subprocess.Popen(
            [python_exe, monitor_script, str(port), reference, info],
            cwd=SCRIPTS_DIR, stdout=self.monitor_log, stderr=subprocess.STDOUT)
        print(f'[Lume] monitor grafici avviato (pid {self.monitor_process.pid}, '
              f'porta {port}); errori in {log_path}')


def add_surface(parent, child_name, mesh_name, color):
    visual = parent.addChild(child_name)
    visual.addObject('MeshOBJLoader', name='loader', filename=os.path.join(MESH_DIR, mesh_name))
    visual.addObject('OglModel', name='visualModel', src='@loader', color=color)
    visual.addObject('RigidMapping')
    return visual


def add_rigid_tube(parent):
    """Parete cilindrica cava; i triangoli interni hanno normale verso l'asse."""
    inner_radius = RIGID_TUBE_INNER_DIAMETER / 2.0
    outer_radius = RIGID_TUBE_OUTER_DIAMETER / 2.0
    positions = []
    for i in range(RIGID_TUBE_AXIAL_SEGMENTS + 1):
        x = RIGID_TUBE_LENGTH * i / RIGID_TUBE_AXIAL_SEGMENTS
        for radius in (inner_radius, outer_radius):
            for j in range(RIGID_TUBE_RADIAL_SEGMENTS):
                angle = 2.0 * math.pi * j / RIGID_TUBE_RADIAL_SEGMENTS
                positions.append([x, radius * math.cos(angle), radius * math.sin(angle)])

    ring = 2 * RIGID_TUBE_RADIAL_SEGMENTS
    n = RIGID_TUBE_RADIAL_SEGMENTS
    triangles = []
    for i in range(RIGID_TUBE_AXIAL_SEGMENTS):
        a, b = i * ring, (i + 1) * ring
        for j in range(n):
            k = (j + 1) % n
            triangles.extend([
                a + j, b + j, b + k,
                a + j, b + k, a + k,
                a + n + j, a + n + k, b + n + k,
                a + n + j, b + n + k, b + n + j,
            ])
    last = RIGID_TUBE_AXIAL_SEGMENTS * ring
    for j in range(n):
        k = (j + 1) % n
        triangles.extend([
            j, n + j, n + k,
            j, n + k, k,
            last + j, last + n + k, last + n + j,
            last + j, last + k, last + n + k,
        ])

    rigid_tube = parent.addChild('TuboEsternoRigido')
    # Visuale: parete completa (superficie interna, esterna e anelli di chiusura).
    wall_visual = rigid_tube.addChild('WallVisual')
    wall_visual.addObject('MeshTopology', name='wallTopology', position=positions,
                          triangles=triangles)
    wall_visual.addObject('OglModel', name='wallVisual', src='@wallTopology',
                          color=[0.35, 0.40, 0.46, 0.35])

    # Collisione: SOLO la superficie interna (normali verso l'asse). Se si usano
    # anche la superficie esterna e gli anelli di chiusura, un nodo che supera la
    # superficie interna viene "agganciato" dalla faccia esterna (a 0.5 mm) e spinto
    # fuori: e' il motivo per cui il tubo attraversava la parete del lume.
    inner_positions, inner_triangles = [], []
    n = RIGID_TUBE_RADIAL_SEGMENTS
    for i in range(RIGID_TUBE_AXIAL_SEGMENTS + 1):
        x = RIGID_TUBE_LENGTH * i / RIGID_TUBE_AXIAL_SEGMENTS
        for j in range(n):
            angle = 2.0 * math.pi * j / n
            inner_positions.append([x, inner_radius * math.cos(angle),
                                    inner_radius * math.sin(angle)])
    for i in range(RIGID_TUBE_AXIAL_SEGMENTS):
        a, b = i * n, (i + 1) * n
        for j in range(n):
            k = (j + 1) % n
            inner_triangles.extend([a + j, b + j, b + k, a + j, b + k, a + k])
    wall_collision = rigid_tube.addChild('InnerWallCollision')
    wall_collision.addObject('MechanicalObject', name='wallDOF', template='Vec3d',
                             position=inner_positions, showObject=False)
    wall_collision.addObject('MeshTopology', name='wallTopology', triangles=inner_triangles)
    # L'interno del lume e' convesso: un segmento tra due punti interni resta
    # interno, quindi basta il contatto punti del tubo / triangoli della parete
    # (Line e Point sulla parete moltiplicano i vincoli senza aggiungere nulla).
    wall_collision.addObject('TriangleCollisionModel', moving=False, simulated=False, group=1)
    return rigid_tube


def createScene(root):
    root.gravity.value = [0.0, 0.0, 0.0]
    root.dt.value = SIMULATION_DT
    root.addObject('RequiredPlugin', pluginName=[
        'Sofa.Component.AnimationLoop', 'Sofa.Component.Collision.Geometry',
        'Sofa.Component.Collision.Detection.Algorithm',
        'Sofa.Component.Collision.Detection.Intersection',
        'Sofa.Component.Collision.Response.Contact',
        'Sofa.Component.Constraint.Lagrangian.Correction',
        'Sofa.Component.Constraint.Lagrangian.Solver',
        'Sofa.Component.Constraint.Projective', 'Sofa.Component.IO.Mesh',
        'Sofa.Component.Engine.Select',
        'Sofa.Component.LinearSolver.Direct',
        'Sofa.Component.LinearSolver.Iterative', 'Sofa.Component.Mapping.Linear',
        'Sofa.Component.Mapping.NonLinear', 'Sofa.Component.Mass',
        'Sofa.Component.ODESolver.Backward',
        'Sofa.Component.SolidMechanics.FEM.Elastic', 'Sofa.Component.SolidMechanics.Spring',
        'Sofa.Component.StateContainer', 'Sofa.Component.Topology.Container.Constant',
        'Sofa.Component.Topology.Container.Dynamic',
        'Sofa.Component.Visual', 'Sofa.GL.Component.Rendering3D', 'Sofa.GL.Component.Shader',
        'SoftRobots', 'STLIB'])
    root.addObject('FreeMotionAnimationLoop')
    constraint_solver = root.addObject('BlockGaussSeidelConstraintSolver',
                                       name='constraintSolver', maxIterations=1000,
                                       tolerance=1e-6, computeConstraintForces=True)
    root.addObject('CollisionPipeline', depth=6)
    root.addObject('BruteForceBroadPhase')
    root.addObject('BVHNarrowPhase')
    root.addObject('RuleBasedContactManager', name='ContactResponse',
                   response='FrictionContactConstraint',
                   responseParams=f'mu={CONTACT_FRICTION}')
    root.addObject('MinProximityIntersection', name='proximity',
                   alarmDistance=CONTACT_ALARM_DISTANCE, contactDistance=CONTACT_DISTANCE)
    root.addObject('VisualStyle', displayFlags='showVisualModels showBehaviorModels')
    root.addObject('LightManager', ambient=[0.55, 0.55, 0.55, 1.0])
    root.addObject('DirectionalLight', direction=[0.4, -0.8, -1.0], color=[0.9, 0.9, 0.9, 1.0])

    # Motore cinematico (posa imposta dal controller) e molla: solo visuale,
    # la legge della molla in serie e' applicata dal controller al CableConstraint.
    motor_node = root.addChild('Motor')
    motor = motor_node.addObject('MechanicalObject', name='motorDOF', template='Rigid3d',
                                 position=[[MOTOR_REST_X, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    motor_visual_node = root.addChild('MotorVisualAnchor')
    motor_visual = motor_visual_node.addObject(
        'MechanicalObject', name='visualDOF', template='Rigid3d',
        position=[[MOTOR_REST_X, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    motor_vis = motor_visual_node.addChild('VisualMotor')
    motor_vis.addObject('MeshOBJLoader', name='loader',
                        filename=os.path.join(MESH_DIR, 'sphere.obj'),
                        scale3d=[MOTOR_SPHERE_RADIUS] * 3)
    motor_vis.addObject('OglModel', name='visualModel', src='@loader',
                        color=[0.62, 0.20, 0.82, 1.0])
    motor_vis.addObject('RigidMapping')

    anchor_node = root.addChild('TendonAnchor')
    anchor = anchor_node.addObject('MechanicalObject', name='anchorDOF', template='Rigid3d',
                                   position=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    add_surface(anchor_node, 'TendonAlongShaft', 'tendon_shaft.obj', [0.95, 0.24, 0.20, 1.0])

    tube = ElasticMaterialObject(
        name='FlexibleTubeFEM',
        volumeMeshFileName=os.path.join(MESH_DIR, 'tube_tetra.vtu'),
        surfaceMeshFileName=os.path.join(MESH_DIR, 'tube_surface.obj'),
        poissonRatio=POISSON_RATIO, youngModulus=E_LOAD_MPA, totalMass=0.76,
        surfaceColor=[0.12, 0.69, 0.82, 1.0], solverName='iterative', withConstrain=True)
    root.addChild(tube)
    if not FREE_MOTION:
        # Superficie di collisione del tubo: punti della superficie esterna mappati
        # sul FEM. Contatto punti-tubo / triangoli-parete interna del lume.
        tube_collision = tube.addChild('TubeCollision')
        tube_collision.addObject('MeshOBJLoader', name='loader',
                                 filename=os.path.join(MESH_DIR, 'tube_surface.obj'))
        tube_collision.addObject('MeshTopology', src='@loader')
        tube_collision.addObject('MechanicalObject', name='collisionDOF')
        tube_collision.addObject('PointCollisionModel', group=2)
        tube_collision.addObject('BarycentricMapping')
    tube.addObject('EulerImplicitSolver', name='odesolver',
                   rayleighStiffness=RAYLEIGH_STIFFNESS, rayleighMass=RAYLEIGH_MASS)
    tube.addObject('SparseLDLSolver', name='linearsolver',
                   template='CompressedRowSparseMatrixMat3x3d')
    FixedBox(applyTo=tube, atPositions=[-0.1, -1.1, -1.1, RIGID_LENGTH + 0.1, 1.1, 1.1],
             name='FixedRigidShaft', doVisualization=False)
    if not FREE_MOTION:
        add_rigid_tube(root)

    cable_points = [[x, TENDON_OFFSET, 0.0] for x in (30.0, 45.0, 60.0, TUBE_LENGTH)]
    cable_node = tube.addChild('TendonCable')
    cable_dofs = cable_node.addObject('MechanicalObject', name='cablePoint', template='Vec3',
                                      position=cable_points)
    cable = cable_node.addObject('CableConstraint', name='tendon',
                                 indices=list(range(len(cable_points))),
                                 pullPoint=CABLE_PULL_POINT, valueType='force', value=[0.0],
                                 minForce=0.0, color=[0.95, 0.24, 0.20, 1.0])
    cable_node.addObject('BarycentricMapping', name='cableMapping')

    cable_visual_node = tube.addChild('TendonVisualGuide')
    cable_visual_node.addObject('EdgeSetTopologyContainer', name='topology',
                                edges=[[i, i + 1] for i in range(len(cable_points) - 1)])
    cable_visual_node.addObject('MechanicalObject', name='tendonGuide', template='Vec3',
                                position=cable_points, showObject=True, showObjectScale=1.4,
                                showColor=[0.95, 0.24, 0.20, 1.0])
    cable_visual_node.addObject('BarycentricMapping', name='visualMapping')

    distal = root.addChild('TubeDistalTip')
    tip = distal.addObject('MechanicalObject', name='tipDOF', template='Rigid3d',
                           position=[[TUBE_LENGTH, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                           showObject=True, showObjectScale=2.0)

    spring_points = helix([MOTOR_REST_X, 0.0, 0.0], [0.0, 0.0, 0.0])
    spring_node = root.addChild('VisualSpringPoints')
    spring_node.addObject('EdgeSetTopologyContainer', name='springPointsTopology',
                          edges=[[i, i + 1] for i in range(len(spring_points) - 1)])
    spring_visual = spring_node.addObject('MechanicalObject', name='springPoints',
                                          template='Vec3d', position=spring_points,
                                          showObject=False)
    spring_wire = spring_node.addChild('ContinuousSpringWire')
    spring_wire.addObject('OglModel', name='springWireModel', src='@../springPointsTopology',
                          lineWidth=5.5, color=[0.62, 0.20, 0.82, 1.0])
    spring_wire.addObject('IdentityMapping')

    centerline_nodes = [list(range(i * TUBE_NODES_PER_SECTION, (i + 1) * TUBE_NODES_PER_SECTION))
                        for i in range(TUBE_AXIAL_CELLS + 1)]
    controller = LumenController(
        name='lumenController', constraintSolver=constraint_solver, motor=motor,
        motorVisual=motor_visual, anchor=anchor, tip=tip, springVisual=spring_visual,
        tube=tube, cable=cable, cableDofs=cable_dofs, centerlineNodes=centerline_nodes)
    root.addObject(controller)
    controller.move_motor()
    controller.launch_live_monitor()

    clearance = RIGID_TUBE_INNER_DIAMETER / 2.0 - OUTER_DIAMETER / 2.0
    print(f'[Lume] caso {run_tag()}: tubo {TUBE_LENGTH:.0f} mm ({RIGID_LENGTH:.0f} incastrati + '
          f'{FLEX_LENGTH:.0f} flessibili), D_lume={RIGID_TUBE_INNER_DIAMETER:g} mm, '
          f'mu={CONTACT_FRICTION:g}, k={SPRING_K:g} N/mm, T_max={MAX_TENDON_FORCE:g} N')
    if not FREE_MOTION:
        print(f'[Lume] gioco radiale tubo-lume = {clearance:.3f} mm '
              f'(DeltaY massimo ammesso ~ {clearance - CONTACT_DISTANCE:.3f} mm)')
    return root
