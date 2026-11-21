# -*- coding: utf-8 -*-
"""Scena SOFA dimostrativa: motore - molla - tubo flessibile in lume rigido.

Avvio da SOFA v26.06.00: runSofa -l SofaPython3 <percorso>\\scena_motore_molla_tubo_base.py
L'attuazione del tendine cresce automaticamente fino a 10 N; 0: reset.

Il tubo esterno rigido e' lungo 100 mm e ha un diametro interno di 3 mm.
Il tendine FEM pieno contenuto e' lungo 100 mm. Unita' geometriche: mm.
"""
import math
import os
import sys
import json
import shutil
import socket
import subprocess
import numpy as np
import Sofa
import Sofa.Core
from Sofa.constants import Key

# I componenti STLIB/SoftRobots sono distribuiti come moduli Python dei plugin SOFA.
SOFA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for plugin_name in ('STLIB', 'SoftRobots'):
    plugin_python = os.path.join(SOFA_ROOT, 'plugins', plugin_name,
                                 'lib', 'python3', 'site-packages')
    if os.path.isdir(plugin_python) and plugin_python not in sys.path:
        sys.path.append(plugin_python)
from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.constraints import FixedBox

TUBE_LENGTH = 100.0
RIGID_TUBE_LENGTH = 100.0
RIGID_LENGTH = 20.0
FLEX_LENGTH = TUBE_LENGTH
OUTER_DIAMETER = 1.93
WALL_THICKNESS = 0.22
INNER_DIAMETER = OUTER_DIAMETER - 2.0 * WALL_THICKNESS
def environment_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError as exc:
        raise RuntimeError(f'{name} deve essere numerico.') from exc


FREE_MOTION = os.environ.get('SOFA_FREE_MOTION', '0') == '1'
RIGID_TUBE_INNER_DIAMETER = environment_float('SOFA_LUMEN_DIAMETER', 3.0)
RIGID_TUBE_WALL_THICKNESS = 0.5
RIGID_TUBE_OUTER_DIAMETER = (RIGID_TUBE_INNER_DIAMETER
                              + 2.0 * RIGID_TUBE_WALL_THICKNESS)
LUMEN_CONTACT_CLEARANCE = 0.02
CONTACT_PENALTY_STIFFNESS = 100.0
CONTACT_PENALTY_DAMPING = 2.0
CONTACT_FRICTION = environment_float('SOFA_FRICTION', 0.1)
CONTACT_RELEASE_DISTANCE = 0.05
CONTACT_FORCE_RELAXATION = 0.995
RIGID_TUBE_RADIAL_SEGMENTS = 32
RIGID_TUBE_AXIAL_SEGMENTS = 100
TENDON_OFFSET = 0.6
SPRING_REST = 20.0
SPRING_K = environment_float('SOFA_SPRING_K', 1.2)
SPRING_DAMPING = 1.5
SPRING_HYSTERESIS = 0.45
E_LOAD_GPA = 77.0
E_UNLOAD_GPA = 65.0
POISSON_RATIO = 0.33
MATERIAL_HYSTERESIS = 0.40
MOTOR_FEED_RATE = 0.35
MAX_MOTOR_OFFSET = environment_float('SOFA_TARGET_OFFSET', 10.0)
MAX_TENDON_FORCE = 10.0
TENDON_FORCE_RAMP_RATE = MAX_TENDON_FORCE / max(MAX_MOTOR_OFFSET / MOTOR_FEED_RATE, 1e-9)
HOLD_TIME = environment_float('SOFA_HOLD_TIME', 0.5)
SIMULATION_DT = 0.01
CONTACT_RAYLEIGH_STIFFNESS = 0.05
CONTACT_RAYLEIGH_MASS = 0.08
MOTOR_SPHERE_RADIUS = 2.2
N_COILS = 9
N_SPRING_POINTS = 181
TUBE_AXIAL_CELLS = 15
TUBE_RADIAL_CELLS = 2
TUBE_CIRCUMFERENTIAL_CELLS = 8
TUBE_NODES_PER_SECTION = TUBE_CIRCUMFERENTIAL_CELLS + 1
MOTOR_REST_X = -SPRING_REST
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, 'scripts')
MESH_DIR = os.path.join(PROJECT_DIR, 'mesh')


def helix(p1, p2):
    """Punti della spirale parametrica, con estremi sul motore e sul tubo."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    axis = p2 - p1
    length = np.linalg.norm(axis)
    if length < 1e-9:
        axis = np.array([1.0, 0.0, 0.0])
    unit = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 0.0, 1.0])
    u = np.cross(unit, ref)
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


class HookSceneController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motor = kwargs['motor']
        self.motor_visual = kwargs['motorVisual']
        self.anchor = kwargs['anchor']
        self.tip = kwargs['tip']
        self.spring_vis = kwargs['springVisual']
        self.tube = kwargs['tube']
        self.tube_dofs = self.tube.dofs
        self.fem_ff = self.tube.forcefield
        self.cable = kwargs['cable']
        self.spring_ff = kwargs['springFF']
        self.spring_k = SPRING_K
        self.centerline_nodes = kwargs['centerlineNodes']
        self.offset = 0.0
        self.applied_tension = 0.0
        self.direction = 0
        self.simulation_time = 0.0
        self.initial_tip_y = None
        self.contact_forces = None
        self.contact_active = [False] * len(self.centerline_nodes)
        self.previous_offset = 0.0
        self.hysteresis_energy = 0.0
        self.phase = 'loading'
        self.phase_time = 0.0
        self.monitor_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.monitor_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.monitor_server.bind(('127.0.0.1', 0))
        self.monitor_server.listen(1)
        self.monitor_server.setblocking(False)
        self.monitor_client = None

    def enforce_lumen_contact(self):
        positions = np.array(self.tube_dofs.position.value, dtype=float, copy=True)
        if positions.ndim != 2:
            return

        if FREE_MOTION:
            self.contact_forces = np.zeros_like(positions)
            return

        velocity = np.array(self.tube_dofs.velocity.value, dtype=float, copy=True)
        previous_contact_forces = (
            np.zeros_like(positions)
            if self.contact_forces is None
            else self.contact_forces.copy()
        )
        self.contact_forces = np.zeros_like(positions)
        for section_index, nodes in enumerate(self.centerline_nodes):
            section = positions[nodes]
            if len(section) == 0 or float(np.mean(section[:, 0])) < RIGID_LENGTH:
                continue
            center = np.mean(section, axis=0)
            radial = center[1:]
            radial_norm = float(np.linalg.norm(radial))
            section_radius = float(np.max(np.linalg.norm(section[:, 1:] - radial, axis=1)))
            allowed_center_radius = (
                RIGID_TUBE_INNER_DIAMETER / 2.0
                - section_radius
                - LUMEN_CONTACT_CLEARANCE
            )
            if radial_norm < 1e-12:
                continue
            if radial_norm <= allowed_center_radius:
                if radial_norm < allowed_center_radius - CONTACT_RELEASE_DISTANCE:
                    self.contact_active[section_index] = False
                if not self.contact_active[section_index]:
                    continue
            else:
                self.contact_active[section_index] = True
            normal = radial / radial_norm
            penetration = radial_norm - allowed_center_radius
            radial_velocity = float(np.mean(velocity[nodes, 1:] @ normal))
            previous_reaction = float(np.linalg.norm(
                previous_contact_forces[nodes].sum(axis=0)))
            reaction = max(
                CONTACT_PENALTY_STIFFNESS * max(0.0, penetration)
                + CONTACT_PENALTY_DAMPING * max(0.0, radial_velocity),
                previous_reaction * CONTACT_FORCE_RELAXATION
            )
            self.contact_forces[nodes, 1:] = (
                -normal * reaction / len(nodes)
            )
            correction = (
                -penetration * normal
            )
            positions[nodes, 1:] += correction
            velocity[nodes, 1:] = 0.0

        self.tube_dofs.position.value = positions.tolist()
        self.tube_dofs.velocity.value = velocity.tolist()

    def move_motor(self):
        x = MOTOR_REST_X - self.offset
        pose = [[x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        # Il cursore e' cinematico, non bloccato da FixedProjectiveConstraint:
        # reimpostiamo posa e velocita' per imporre la corsa del motore.
        self.motor.position.value = pose
        self.motor.velocity.value = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        # La posa visuale non riceve la forza della molla: sfera e spirale
        # restano quindi solidali anche dopo l'integrazione del solver.
        self.motor_visual.position.value = pose
        self.update_spring_points()

    def update_spring_points(self):
        p1 = self.motor_visual.position.value[0][:3]
        p2 = self.anchor.position.value[0][:3]
        self.spring_vis.position.value = helix(p1, p2)

    def centerline(self):
        pos = np.asarray(self.tube_dofs.position.value, dtype=float)
        # Prima di Simulation.init il prefab STLIB espone ancora un placeholder
        # a un solo nodo: in quel momento la linea centrale è quella indeformata.
        last_node = max(nodes[-1] for nodes in self.centerline_nodes)
        if pos.ndim != 2 or len(pos) <= last_node:
            return np.asarray([[x, 0.0, 0.0]
                               for x in np.linspace(0.0, TUBE_LENGTH,
                                                    len(self.centerline_nodes))])
        return np.asarray([np.mean(pos[nodes], axis=0) for nodes in self.centerline_nodes])

    def update_hook(self):
        """Aggiorna il marker di punta a partire dalla deformazione FEM reale."""
        line = self.centerline()
        tip_pos = line[-1]
        self.tip.position.value = [[float(tip_pos[0]), float(tip_pos[1]), float(tip_pos[2]),
                                    0.0, 0.0, 0.0, 1.0]]

    def spring_elongation(self):
        p1 = np.asarray(self.motor.position.value[0][:3], dtype=float)
        p2 = np.asarray(self.anchor.position.value[0][:3], dtype=float)
        return max(0.0, float(np.linalg.norm(p2 - p1)) - SPRING_REST)

    def send_live_sample(self):
        pos = self.tip.position.value[0]
        x_tip, y_tip = float(pos[0]), float(pos[1])
        if self.initial_tip_y is None:
            self.initial_tip_y = y_tip
        delta_y = y_tip - self.initial_tip_y
        delta_ls = self.spring_elongation()
        displacement_increment = abs(self.offset - self.previous_offset)
        contact_magnitude = float(np.linalg.norm(self.contact_forces, axis=1).sum())
        self.hysteresis_energy += (
            CONTACT_FRICTION * contact_magnitude * displacement_increment
        )
        self.previous_offset = self.offset
        positions = np.asarray(self.tube_dofs.position.value, dtype=float)
        if self.contact_forces is None:
            raise RuntimeError('Forze di contatto non inizializzate.')
        if self.monitor_client is None:
            try:
                self.monitor_client, _ = self.monitor_server.accept()
            except BlockingIOError:
                return
        profile = [
            [float(position[0]), float(np.linalg.norm(force[:3]))]
            for position, force in zip(positions, self.contact_forces)
        ]
        payload = {
            'time': self.simulation_time,
            'u': self.offset,
            'delta_y': delta_y,
            'hysteresis': self.hysteresis_energy,
            'tension': self.applied_tension,
            'target_reached': self.offset >= MAX_MOTOR_OFFSET,
            'phase': self.phase,
            'diameter': RIGID_TUBE_INNER_DIAMETER,
            'friction': CONTACT_FRICTION,
            'spring_k': self.spring_k,
            'free_motion': FREE_MOTION,
            'contact': profile,
        }
        try:
            self.monitor_client.sendall(
                (json.dumps(payload) + '\n').encode('utf-8')
            )
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
            self.monitor_client.close()
            self.monitor_client = None

    def launch_live_monitor(self):
        python_exe = shutil.which('pythonw.exe') or shutil.which('python.exe')
        if python_exe is None:
            raise RuntimeError('Interprete Python non trovato per il monitor live.')
        monitor_script = os.path.join(SCRIPTS_DIR, 'plot_live.py')
        port = self.monitor_server.getsockname()[1]
        self.monitor_process = subprocess.Popen(
            [python_exe, monitor_script, str(port)],
            cwd=os.path.dirname(monitor_script)
        )

    def report(self):
        delta_ls = self.offset
        motor_x = MOTOR_REST_X - self.offset
        pos = self.tip.position.value[0]
        print(f"[Motore-molla-tubo] u={self.offset:.2f} mm, xmot={motor_x:.2f} mm | "
              f"k={self.spring_k:.1f} N/mm | T={self.applied_tension:.3f} N | "
              f"DeltaLs={delta_ls:.2f} mm | punta X={pos[0]:.2f}, Y={pos[1]:.2f} mm")

    def onKeypressedEvent(self, event):
        key = event.get('key', '')
        if key in ('0', 'z', 'Z'):
            self.offset = 0.0
            self.applied_tension = 0.0
            self.direction = 0
            self.cable.value.value = [0.0]
            self.previous_offset = 0.0
            self.hysteresis_energy = 0.0
            self.phase = 'loading'
            self.phase_time = 0.0
        else:
            return
        self.report()

    def compute_tension(self, delta_ls):
        hysteresis = (SPRING_HYSTERESIS + MATERIAL_HYSTERESIS) * self.direction
        # Legge del paper: la tensione dipende dall'allungamento e dall'isteresi;
        # il termine viscoso non viene aggiunto al comando del CableConstraint.
        raw_tension = self.spring_k * delta_ls + hysteresis
        return max(0.0, raw_tension)

    def update_mechanics(self):
        p1 = np.asarray(self.motor.position.value[0][:3], dtype=float)
        p2 = np.asarray(self.anchor.position.value[0][:3], dtype=float)
        delta_ls = max(0.0, float(np.linalg.norm(p2 - p1)) - SPRING_REST)
        self.direction = 1
        self.cable.value.value = [self.applied_tension]
        self.fem_ff.youngModulus.value = [E_LOAD_GPA * 1000.0]
        self.update_hook()

    def onAnimateBeginEvent(self, event):
        self.dt = float(event.get('dt', 0.01))
        self.phase_time += self.dt
        if self.phase == 'loading':
            self.offset = min(MAX_MOTOR_OFFSET,
                              self.offset + MOTOR_FEED_RATE * self.dt)
            if self.offset >= MAX_MOTOR_OFFSET:
                self.phase = 'holding'
                self.phase_time = 0.0
        elif self.phase == 'holding' and self.phase_time >= HOLD_TIME:
            self.phase = 'unloading'
            self.phase_time = 0.0
        elif self.phase == 'unloading':
            self.offset = max(0.0, self.offset - MOTOR_FEED_RATE * self.dt)
            if self.offset <= 0.0:
                self.phase = 'complete'
                self.phase_time = 0.0

        self.applied_tension = MAX_TENDON_FORCE * (
            self.offset / max(MAX_MOTOR_OFFSET, 1e-9)
        )
        self.direction = 1 if self.phase == 'loading' else -1
        # Mantiene il carrello sulla posizione comandata anche durante la
        # dinamica, senza lasciare che la forza della molla lo trascini.
        self.move_motor()
        self.update_mechanics()
        self.update_spring_points()

    def onAnimateEndEvent(self, event):
        self.enforce_lumen_contact()
        self.update_hook()
        self.update_spring_points()
        self.simulation_time += float(event.get('dt', getattr(self, 'dt', 0.01)))
        self.send_live_sample()

    def onSimulationEndEvent(self, event):
        if self.monitor_client is not None:
            self.monitor_client.close()
        self.monitor_server.close()


def add_surface(parent, child_name, mesh_name, color):
    visual = parent.addChild(child_name)
    visual.addObject('MeshOBJLoader', name='loader',
                     filename=os.path.join(MESH_DIR, mesh_name))
    visual.addObject('OglModel', name='visualModel', src='@loader',
                     color=color)
    visual.addObject('RigidMapping')
    return visual


def add_rigid_tube(parent):
    """Crea una parete cilindrica cava con estremita' anulari chiuse."""
    inner_radius = RIGID_TUBE_INNER_DIAMETER / 2.0
    outer_radius = RIGID_TUBE_OUTER_DIAMETER / 2.0
    positions = []
    for i in range(RIGID_TUBE_AXIAL_SEGMENTS + 1):
        x = RIGID_TUBE_LENGTH * i / RIGID_TUBE_AXIAL_SEGMENTS
        for radius in (inner_radius, outer_radius):
            for j in range(RIGID_TUBE_RADIAL_SEGMENTS):
                angle = 2.0 * math.pi * j / RIGID_TUBE_RADIAL_SEGMENTS
                positions.append([
                    x,
                    radius * math.cos(angle),
                    radius * math.sin(angle)
                ])

    ring_size = 2 * RIGID_TUBE_RADIAL_SEGMENTS
    triangles = []
    for i in range(RIGID_TUBE_AXIAL_SEGMENTS):
        first = i * ring_size
        next_ring = (i + 1) * ring_size
        for j in range(RIGID_TUBE_RADIAL_SEGMENTS):
            next_j = (j + 1) % RIGID_TUBE_RADIAL_SEGMENTS
            inner = first + j
            inner_next = first + next_j
            inner_next_ring = next_ring + j
            inner_next_ring_next = next_ring + next_j
            outer = first + RIGID_TUBE_RADIAL_SEGMENTS + j
            outer_next = first + RIGID_TUBE_RADIAL_SEGMENTS + next_j
            outer_next_ring = next_ring + RIGID_TUBE_RADIAL_SEGMENTS + j
            outer_next_ring_next = next_ring + RIGID_TUBE_RADIAL_SEGMENTS + next_j
            triangles.extend([
                inner, inner_next_ring, inner_next_ring_next,
                inner, inner_next_ring_next, inner_next,
                outer, outer_next, outer_next_ring_next,
                outer, outer_next_ring_next, outer_next_ring
            ])

    # Chiude la parete alle estremita' senza chiudere il lume interno.
    first_inner = 0
    first_outer = RIGID_TUBE_RADIAL_SEGMENTS
    last_ring = RIGID_TUBE_AXIAL_SEGMENTS * ring_size
    last_inner = last_ring
    last_outer = last_ring + RIGID_TUBE_RADIAL_SEGMENTS
    for j in range(RIGID_TUBE_RADIAL_SEGMENTS):
        next_j = (j + 1) % RIGID_TUBE_RADIAL_SEGMENTS
        triangles.extend([
            first_inner + j, first_outer + j, first_outer + next_j,
            first_inner + j, first_outer + next_j, first_inner + next_j,
            last_inner + j, last_outer + next_j, last_outer + j,
            last_inner + j, last_inner + next_j, last_outer + next_j
        ])

    rigid_tube = parent.addChild('TuboEsternoRigido')
    rigid_tube.addObject('MechanicalObject', name='wallDOF', template='Vec3d',
                         position=positions, showObject=False)
    rigid_tube.addObject('MeshTopology', name='wallTopology',
                         triangles=triangles)
    rigid_tube.addObject('PointCollisionModel', name='wallPointCollision',
                         moving=False, simulated=False)
    rigid_tube.addObject('LineCollisionModel', name='wallLineCollision',
                         moving=False, simulated=False)
    rigid_tube.addObject('TriangleCollisionModel', name='wallCollision',
                         moving=False, simulated=False,
                         contactDistance=0.01)
    wall_visual = rigid_tube.addChild('WallVisual')
    wall_visual.addObject('OglModel', name='wallVisual',
                          src='@../wallTopology',
                          color=[0.35, 0.40, 0.46, 1.0])
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
        'Sofa.Component.Mapping.NonLinear', 'Sofa.Component.Mass', 'Sofa.Component.ODESolver.Backward',
        'Sofa.Component.SolidMechanics.FEM.Elastic', 'Sofa.Component.SolidMechanics.Spring',
        'Sofa.Component.StateContainer', 'Sofa.Component.Topology.Container.Constant',
        'Sofa.Component.Topology.Container.Dynamic',
        'Sofa.Component.Visual', 'Sofa.GL.Component.Rendering3D', 'Sofa.GL.Component.Shader',
        'SoftRobots', 'STLIB'])
    root.addObject('FreeMotionAnimationLoop')
    root.addObject(
        'BlockGaussSeidelConstraintSolver',
        maxIterations=1000,
        tolerance=1e-6,
        computeConstraintForces=True
    )
    root.addObject('CollisionPipeline', depth=6)
    root.addObject('BruteForceBroadPhase')
    root.addObject('BVHNarrowPhase')
    root.addObject(
        'RuleBasedContactManager',
        name='ContactResponse',
        response='FrictionContactConstraint',
        responseParams=f'mu={CONTACT_FRICTION}'
    )
    root.addObject(
        'LocalMinDistance',
        alarmDistance=0.30,
        contactDistance=0.08,
        angleCone=0.01
    )
    root.addObject('VisualStyle', displayFlags='showVisualModels showBehaviorModels')
    root.addObject('LightManager', ambient=[0.55, 0.55, 0.55, 1.0])
    root.addObject('DirectionalLight', direction=[0.4, -0.8, -1.0], color=[0.9, 0.9, 0.9, 1.0])

    motor_node = root.addChild('Motor')
    motor = motor_node.addObject('MechanicalObject', name='motorDOF', template='Rigid3d',
        position=[[MOTOR_REST_X, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    motor_node.addObject('UniformMass', totalMass=1e-6)
    motor_visual_node = root.addChild('MotorVisualAnchor')
    motor_visual = motor_visual_node.addObject('MechanicalObject', name='visualDOF',
        template='Rigid3d', position=[[MOTOR_REST_X, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    motor_visual_node.addObject('UniformMass', totalMass=1e-9)
    motor_vis = motor_visual_node.addChild('VisualMotor')
    motor_vis.addObject('MeshOBJLoader', name='loader', filename=os.path.join(MESH_DIR, 'sphere.obj'),
        scale3d=[MOTOR_SPHERE_RADIUS] * 3)
    motor_vis.addObject('OglModel', name='visualModel', src='@loader', color=[0.62, 0.20, 0.82, 1.0])
    motor_vis.addObject('RigidMapping')

    anchor_node = root.addChild('TendonAnchor')
    anchor = anchor_node.addObject('MechanicalObject', name='anchorDOF', template='Rigid3d',
        position=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    anchor_node.addObject('FixedProjectiveConstraint', indices=[0])
    motor_spring = root.addObject('SpringForceField', name='motorSpring', object1='@Motor/motorDOF',
        object2='@TendonAnchor/anchorDOF', spring=[0, 0, SPRING_K, SPRING_DAMPING, SPRING_REST])
    add_surface(anchor_node, 'TendonAlongShaft', 'tendon_shaft.obj', [0.95, 0.24, 0.20, 1.0])

    # Struttura d'esempio 4: prefab STLIB, mesh tetraedrica, solver iterativo,
    # TetrahedronFEMForceField e superficie visuale/collisione mappate sul volume.
    volume_path = os.path.join(MESH_DIR, 'tube_tetra.vtu')
    surface_path = os.path.join(MESH_DIR, 'tube_surface.obj')
    tube = ElasticMaterialObject(name='FlexibleTubeFEM', volumeMeshFileName=volume_path,
        surfaceMeshFileName=surface_path, collisionMesh=surface_path,
        poissonRatio=POISSON_RATIO, youngModulus=E_LOAD_GPA * 1000.0,
        totalMass=0.76, surfaceColor=[0.12, 0.69, 0.82, 1.0],
        solverName='iterative', withConstrain=True)
    root.addChild(tube)
    tube.addObject(
        'EulerImplicitSolver',
        name='odesolver',
        rayleighStiffness=CONTACT_RAYLEIGH_STIFFNESS,
        rayleighMass=CONTACT_RAYLEIGH_MASS
    )
    # Il solver diretto fornisce a LinearSolverConstraintCorrection la compliance
    # FEM reale, necessaria per trasformare la forza del CableConstraint in flessione.
    tube.addObject('SparseLDLSolver', name='linearsolver',
        template='CompressedRowSparseMatrixMat3x3d')
    FixedBox(applyTo=tube, atPositions=[-0.1, -1.1, -1.1, RIGID_LENGTH + 0.1, 1.1, 1.1],
             name='FixedRigidShaft', doVisualization=False)
    if not FREE_MOTION:
        add_rigid_tube(root)

    # Tendine guidato sul lato +Y con punti di passaggio lungo il FEM. I punti
    # trasferiscono reazioni distribuite al tubo, evitando una trazione quasi
    # assiale concentrata solo sulla punta. Il primo è oltre l'incastro a x=20.
    # Il cavo meccanico è incorporato nella sezione piena del tendine.
    cable_guide_y = TENDON_OFFSET
    cable_points = [[x, cable_guide_y, 0.0] for x in (30.0, 45.0, 60.0, TUBE_LENGTH)]
    cable_node = tube.addChild('TendonCable')
    cable_node.addObject('MechanicalObject', name='cablePoint', template='Vec3',
        position=cable_points)
    cable = cable_node.addObject('CableConstraint', name='tendon',
        indices=list(range(len(cable_points))),
        pullPoint=[RIGID_LENGTH, TENDON_OFFSET, 0.0], valueType='force', value=[0.0],
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
        position=[[TUBE_LENGTH, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    distal.addObject('UniformMass', totalMass=1e-9)

    spring_points = helix([MOTOR_REST_X, 0.0, 0.0], [0.0, 0.0, 0.0])
    spring_node = root.addChild('VisualSpringPoints')
    spring_node.addObject('EdgeSetTopologyContainer', name='springPointsTopology',
        edges=[[i, i + 1] for i in range(len(spring_points) - 1)])
    spring_visual = spring_node.addObject('MechanicalObject', name='springPoints', template='Vec3d',
        position=spring_points, showObject=False)
    spring_wire = spring_node.addChild('ContinuousSpringWire')
    spring_wire.addObject('OglModel', name='springWireModel', src='@../springPointsTopology',
        lineWidth=5.5, color=[0.62, 0.20, 0.82, 1.0])
    spring_wire.addObject('IdentityMapping')

    # L'indice viene tenuto per riferimento: la posizione marker segue la mesh deformata.
    centerline_nodes = [list(range(i * TUBE_NODES_PER_SECTION,
                                    (i + 1) * TUBE_NODES_PER_SECTION))
                        for i in range(TUBE_AXIAL_CELLS + 1)]
    controller = HookSceneController(name='hookController', motor=motor, motorVisual=motor_visual,
        anchor=anchor, tip=tip, springVisual=spring_visual, tube=tube, cable=cable,
        springFF=motor_spring,
        centerlineNodes=centerline_nodes)
    root.addObject(controller)
    controller.move_motor()
    controller.update_hook()
    controller.launch_live_monitor()

    print(f"Tendine FEM pieno: {RIGID_LENGTH:.0f} mm rigidi + {FLEX_LENGTH:.0f} mm flessibili; "
          f"diametro={OUTER_DIAMETER:.2f} mm")
    print(f"Lunghezza totale a riposo: {SPRING_REST:.0f} mm molla + {RIGID_LENGTH:.0f} mm albero + "
          f"{FLEX_LENGTH:.0f} mm flessibili = {SPRING_REST + RIGID_LENGTH + FLEX_LENGTH:.0f} mm")
    print(f"Nitinol: E carico/scarico={E_LOAD_GPA:.0f}/{E_UNLOAD_GPA:.0f} GPa, nu={POISSON_RATIO:.2f}; "
          f"molla k={SPRING_K:.1f} N/mm fisso; CableConstraint in forza, percorso guidato +Y, "
          f"offset nominale +{TENDON_OFFSET:.1f} mm")
    print(f"Attuazione automatica del tendine: rampa {TENDON_FORCE_RAMP_RATE:.2f} N/s, "
          f"limite {MAX_TENDON_FORCE:.2f} N.")
    print("Controllo: 0 reset.")
    print("Monitor live: DeltaY-u, profilo forza di contatto e isteresi H.")
    return root
