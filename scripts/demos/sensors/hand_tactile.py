# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Sunghyun Park

"""Press the tactile cells of the AI Worker's HX5-D20 fingertips from a remote control window.

A probe presses the fingertip cell picked on the remote with the force asked for. The cell lights up on the pad and
in the hand's tactile heatmap, and the remote shows its force and its reading from 0 to 255. The arm is held still,
like a hand mounted on a test bench, so only the fingers give way.

Without a window, the demo presses the centre cell of every fingertip once and prints what the sensor read. It
exits with status 1 if a press lit any other cell most or missed the asked force.

.. code-block:: bash

    python scripts/demos/sensors/hand_tactile.py
    python scripts/demos/sensors/hand_tactile.py --headless

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Press the HX5-D20 fingertip tactile cells of an AI Worker (FFW-SH5).")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--headless", action="store_true", help="Press every fingertip once and print the readings.")
# The remote needs the Isaac Sim window, which opens only with the Kit visualizer.
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()
if args_cli.headless:
    args_cli.visualizer = ["none"]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys
from dataclasses import dataclass
from functools import partial

import isaaclab.sim as sim_utils
import numpy as np
import torch
from cyclo_lab.assets.robots.FFW_SH5 import FFW_SH5_CFG
from cyclo_lab.sensors import HandTactile, HandTactileCfg
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_visualizers.kit import KitVisualizerCfg

PHYSICS_DT = 1.0 / 120.0
PROBE_RADIUS = 0.002  # in m
HOVER = 0.002  # gap between the probe and the pad while it is off the pad, in m
SLIDE_SPEED, APPROACH_SPEED = 0.12, 0.01  # probe speeds over the pad and towards it, in m/s
PRESS_GAIN = 0.0003  # how fast the probe sinks for each newton still missing, in m/s/N
CONTACT_FORCE = 0.05  # pad force from which the probe counts as touching, in N
FORCE_TOLERANCE = 0.1  # how far off the asked force a press may read and still count as reached, relative
# On the curved outer cells, the pad force jitters by some 15 % from step to step under a steady press, so a press
# counts as reached on the force averaged over about this long, in s.
FORCE_AVERAGE_TIME = 0.05
MAX_DEPTH = 0.006  # deepest the probe goes past the pad surface, in m
FORCE_RANGE = (0.5, 20.0)  # in N
DEFAULT_FORCE = 3.0
SWEEP_CELLS = (0, 1, 2, 5, 4, 3, 6, 7, 8)  # all nine cells, row by row like a snake
SWEEP_HOLD = 0.3  # how long each cell is held at the asked force during a sweep, in s
CENTER_CELL = 4
FLOOR_HEIGHT = -0.185  # just below the AI Worker's wheels
BACKGROUND = (0.86, 0.87, 0.89)  # a light, neutral grey that the dark AI Worker stands out against
# The thumbs are held a little open. At zero, the second thumb link sits inside the hand base, whose collisions
# FFW_SH5_CFG filters for the other four fingers only, so the thumb jams against the base and jitters.
OPEN_THUMBS = {"finger_l_joint2": 0.6, "finger_r_joint2": -0.6}
# Every joint but the fingers is held very stiffly, like a hand clamped on a test bench, so that a press bends the
# fingers only and the probe reaches the whole force range.
BENCH_ACTUATORS = {
    name: (
        actuator
        if any(joints.startswith("finger_") for joints in actuator.joint_names_expr)
        else actuator.replace(stiffness=1.0e6, damping=1.0e4, effort_limit_sim=1.0e6)
    )
    for name, actuator in FFW_SH5_CFG.actuators.items()
}


@configclass
class TactileDemoSceneCfg(InteractiveSceneCfg):
    """An AI Worker with tactile fingertips and the remote's probe."""

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9), visible_in_primary_ray=False),
    )
    floor = AssetBaseCfg(
        prim_path="/World/Floor",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, FLOOR_HEIGHT)),
    )
    robot = FFW_SH5_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=FFW_SH5_CFG.spawn.replace(activate_contact_sensors=True),
        actuators=BENCH_ACTUATORS,
    )
    # The probe's tip gives a little, 0.2 mm at 20 N, like the rubber tip of a force gauge. With a rigid tip, the
    # contact force on a thumb jumped from step to step.
    probe = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Probe",
        spawn=sim_utils.SphereCfg(
            radius=PROBE_RADIUS,
            rigid_props=sim_utils.PhysxRigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.PhysxCollisionPropertiesCfg(),
            physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
                compliant_contact_stiffness=1.0e5, compliant_contact_damping=300.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 3.0)),
    )
    tactile_left = HandTactileCfg(prim_path="{ENV_REGEX_NS}/Robot", hand="left", debug_vis=True)
    tactile_right = HandTactileCfg(prim_path="{ENV_REGEX_NS}/Robot", hand="right")


@dataclass
class Press:
    """What the remote asks for. The remote only edits it; the simulation loop carries it out."""

    hand: str = "left"
    finger: int = 1
    cell: int = CENTER_CELL
    force: float = DEFAULT_FORCE
    """Force to press with, in N."""
    pressing: bool = False
    sweep: list[int] | None = None
    """Cells still to press in the running sweep, or None when no sweep runs."""
    show_readings: bool = False
    """Whether the heatmap shows each cell's reading, from 0 to 255."""


class Probe:
    """The probe ball. It glides over to a cell, sinks into the pad until the pad reads the asked force, and lifts
    straight off before it moves on."""

    def __init__(self, ball: RigidObject, radius: float):
        self._ball = ball
        self._radius = radius
        self._center = np.zeros(3)
        self._cell = None
        self._pad = None
        """Pad point and outward normal that the probe presses. They stay fixed during a press."""
        self._depth = -HOVER
        """How far past that pad point the probe is pushed, in m. Negative while it is off the pad."""
        self._average_force = 0.0
        self.settled = 0.0
        """How long the pad has read the asked force, in s."""

    def place(self, cell: tuple[int, int], point: np.ndarray, normal: np.ndarray):
        """Puts the probe right above a cell, off the pad."""
        self._center = point + normal * (self._radius + HOVER)
        self._cell, self._pad, self._depth, self.settled, self._average_force = cell, None, -HOVER, 0.0, 0.0
        self._move_ball()

    def step(self, cell, point: np.ndarray, normal: np.ndarray, force: float | None, force_now: float, dt: float):
        """Moves the probe for one physics step.

        Args:
            cell: The (finger, cell) to press.
            point: Where the centre of that cell is on the pad.
            normal: The pad's outward normal at that point.
            force: The force to press with, in N, or None to stop and stay off the pad.
            force_now: The force the pad reads, in N.
            dt: The physics time step, in s.
        """
        if self._pad is not None and (force is None or cell != self._cell):
            self._depth = max(-HOVER, self._depth - APPROACH_SPEED * dt)  # off the pad before anything else
            if self._depth == -HOVER:
                self._pad = None
        elif self._pad is None and force is not None:
            hover = point + normal * (self._radius + HOVER)
            self._center = _toward(self._center, hover, SLIDE_SPEED * dt)
            if np.array_equal(self._center, hover):
                self._cell, self._pad = cell, (point, normal)
        elif self._pad is not None:
            speed = PRESS_GAIN * (force - force_now) if force_now >= CONTACT_FORCE else APPROACH_SPEED
            self._depth = min(self._depth + float(np.clip(speed, -APPROACH_SPEED, APPROACH_SPEED)) * dt, MAX_DEPTH)

        if self._pad is not None:
            pad_point, pad_normal = self._pad
            self._center = pad_point + pad_normal * (self._radius - self._depth)
        self._average_force += min(1.0, dt / FORCE_AVERAGE_TIME) * (force_now - self._average_force)
        # Only a press on the asked cell counts, so the time held starts over when the remote picks another cell.
        reached = (
            force is not None
            and self._pad is not None
            and cell == self._cell
            and abs(self._average_force - force) <= FORCE_TOLERANCE * force
        )
        self.settled = self.settled + dt if reached else 0.0
        self._move_ball()

    def _move_ball(self):
        self._ball.write_root_pose_to_sim_index(root_pose=_pose(self._center, self._ball.device))


class Camera:
    """Glides the viewport camera to a hand, so that all five fingertip pads face it."""

    def __init__(self, sim: sim_utils.SimulationContext, glide_time: float = 0.8):
        self._sim = sim
        self._glide_time = glide_time
        self._view = self._start = self._goal = None
        self._elapsed = 0.0

    def look_at(self, sensor: HandTactile):
        eye, target = _hand_view(sensor)
        self._start, self._goal, self._elapsed = self._view or (eye, target), (eye, target), 0.0

    def step(self, dt: float):
        if self._goal is None or self._view is self._goal:
            return
        self._elapsed += dt
        s = min(1.0, self._elapsed / self._glide_time)
        s = s * s * (3.0 - 2.0 * s)  # ease in and out
        if s >= 1.0:
            self._view = self._goal
        else:
            self._view = tuple(start + s * (goal - start) for start, goal in zip(self._start, self._goal))
        self._sim.set_camera_view(*self._view)


class Remote:
    """The remote control window. Its buttons only edit :class:`Press`."""

    def __init__(self, press: Press, finger_names: tuple[str, ...]):
        import omni.ui as ui

        self._press = press
        self._finger_names = finger_names
        self._window = ui.Window("Tactile Remote", width=340, height=400)
        selected = {"Button:selected": {"background_color": 0xFF9A6A3A}}
        # Labels sit left of their controls, as in the Property panel, so that the whole remote fits without scrolling.
        label_width = 64
        with self._window.frame, ui.VStack(spacing=6, height=0, style=selected):
            with ui.HStack(spacing=4, height=26):
                ui.Label("Hand", width=label_width)
                self._hand_buttons = {
                    hand: ui.Button(hand.capitalize(), clicked_fn=partial(self._pick_hand, hand))
                    for hand in ("left", "right")
                }
            with ui.HStack(spacing=4, height=26):
                ui.Label("Finger", width=label_width)
                self._finger_buttons = [
                    ui.Button(name.capitalize(), clicked_fn=partial(self._pick_finger, finger))
                    for finger, name in enumerate(finger_names)
                ]
            # The buttons sit like the heatmap's cells: the top row is at the fingertip end.
            with ui.HStack(spacing=4, height=0):
                ui.Label("Cell", width=label_width, alignment=ui.Alignment.LEFT_TOP)
                self._cell_buttons = []
                with ui.VStack(spacing=4, width=0):
                    for row in range(3):
                        with ui.HStack(spacing=4, height=28):
                            for column in range(3):
                                cell = 3 * row + column
                                button = ui.Button(str(cell + 1), width=34, clicked_fn=partial(self._press_cell, cell))
                                self._cell_buttons.append(button)
                ui.Spacer(width=8)
                ui.Label("Top row: fingertip end, as in the heatmap.", word_wrap=True, alignment=ui.Alignment.LEFT_TOP)
            with ui.HStack(spacing=4, height=24):
                ui.Label("Force (N)", width=label_width)
                self._force = ui.FloatSlider(min=FORCE_RANGE[0], max=FORCE_RANGE[1])
                self._force.model.set_value(press.force)
                self._force.model.add_value_changed_fn(
                    lambda model: setattr(press, "force", model.get_value_as_float())
                )
            self._reading = ui.Label("", word_wrap=True)
            with ui.HStack(spacing=6, height=22):
                self._show_readings = ui.CheckBox(width=20)
                self._show_readings.model.set_value(press.show_readings)
                self._show_readings.model.add_value_changed_fn(
                    lambda model: setattr(press, "show_readings", model.get_value_as_bool())
                )
                ui.Label("Show each cell's reading (0-255) in the heatmap")
            with ui.HStack(spacing=4, height=28):
                ui.Button(
                    "Press all 9",
                    clicked_fn=self._sweep,
                    tooltip="Press the nine cells of this finger one after another.",
                )
                ui.Button("Lift off", clicked_fn=self._lift, tooltip="Stop and lift the probe off the pad.")
            self._status = ui.Label("", word_wrap=True)
        # A tab beside the Property panel never covers the hand in the viewport.
        self._window.deferred_dock_in("Property", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)

    def refresh(self, pad_force: float, cell_force: float, cell_reading: float):
        """Shows what is selected and what the pad reads."""
        press = self._press
        for hand, button in self._hand_buttons.items():
            _set(button, "selected", hand == press.hand)
        for finger, button in enumerate(self._finger_buttons):
            _set(button, "selected", finger == press.finger)
        for cell, button in enumerate(self._cell_buttons):
            _set(button, "selected", press.pressing and cell == press.cell)
        action = "pressing" if press.pressing else "lifted off"
        finger = self._finger_names[press.finger]
        _set(self._status, "text", f"{press.hand.capitalize()} {finger}, cell {press.cell + 1}: {action}")
        _set(
            self._reading,
            "text",
            f"Cell {press.cell + 1}: {cell_force:.1f} N, reading {cell_reading:.0f} of 255. Pad: {pad_force:.1f} N.",
        )

    def _pick_hand(self, hand: str):
        self._press.hand = hand
        self._lift()

    def _pick_finger(self, finger: int):
        self._press.finger = finger
        self._press_cell(self._press.cell)

    def _press_cell(self, cell: int):
        self._press.cell, self._press.pressing, self._press.sweep = cell, True, None

    def _sweep(self):
        self._press.cell, self._press.pressing, self._press.sweep = SWEEP_CELLS[0], True, list(SWEEP_CELLS[1:])

    def _lift(self):
        self._press.pressing, self._press.sweep = False, None


def run_remote(sim: sim_utils.SimulationContext, scene: InteractiveScene, step):
    """Runs the demo with the remote control window until Isaac Sim is closed."""
    sensors = {"left": scene["tactile_left"], "right": scene["tactile_right"]}
    probe = Probe(scene["probe"], PROBE_RADIUS)
    camera, press = Camera(sim), Press()
    remote = Remote(press, sensors["left"].finger_names)
    shown = readings_shown = None
    while simulation_app.is_running():
        sensor = sensors[press.hand]
        if (press.hand, press.finger) != shown:
            if shown is None or press.hand != shown[0]:
                for hand, hand_sensor in sensors.items():
                    hand_sensor.set_debug_vis(hand == press.hand)
                _place_heatmap(sensor)
                camera.look_at(sensor)
            probe.place((press.finger, press.cell), *sensor.cell_pose_w(press.finger, press.cell))
            shown = (press.hand, press.finger)
        if press.show_readings != readings_shown:
            for hand_sensor in sensors.values():
                hand_sensor.set_heatmap_readings(press.show_readings)
            readings_shown = press.show_readings
        forces = sensor.data.cell_forces.torch[0, press.finger]
        pad_force, cell_force = float(forces.sum()), float(forces[press.cell])
        cell_reading = float(sensor.data.cell_readings.torch[0, press.finger, press.cell])
        point, normal = sensor.cell_pose_w(press.finger, press.cell)
        force = press.force if press.pressing else None
        probe.step((press.finger, press.cell), point, normal, force, pad_force, PHYSICS_DT)
        camera.step(PHYSICS_DT)
        remote.refresh(pad_force, cell_force, cell_reading)
        # After the probe's step, which starts the time held over if the remote has just picked another cell.
        if press.sweep is not None and probe.settled >= SWEEP_HOLD:
            if press.sweep:
                press.cell = press.sweep.pop(0)
            else:
                press.pressing, press.sweep = False, None
        step()


def run_tour(scene: InteractiveScene, step) -> bool:
    """Presses the centre cell of every fingertip with the default force and reports what the sensor read.

    Returns:
        Whether every press lit the pressed cell most and reached the asked force.
    """
    probe, passed = Probe(scene["probe"], PROBE_RADIUS), True
    for hand in ("left", "right"):
        sensor = scene[f"tactile_{hand}"]
        names = sensor.finger_names
        for finger, name in enumerate(names):
            cell = (finger, CENTER_CELL)
            probe.place(cell, *sensor.cell_pose_w(*cell))
            for force, duration in ((DEFAULT_FORCE, 3.0), (None, 0.5)):
                for _ in range(round(duration / PHYSICS_DT)):
                    forces = sensor.data.cell_forces.torch[0].clone()
                    readings = sensor.data.cell_readings.torch[0].clone()
                    probe.step(cell, *sensor.cell_pose_w(*cell), force, float(forces[finger].sum()), PHYSICS_DT)
                    step()
                    if force is not None and probe.settled >= SWEEP_HOLD:
                        break
                if force is not None:
                    pressed, pressed_readings = forces, readings
            lit = divmod(int(pressed.argmax()), pressed.shape[1])
            pad_force = float(pressed[finger].sum())
            correct = lit == cell and abs(pad_force - DEFAULT_FORCE) <= FORCE_TOLERANCE * DEFAULT_FORCE
            passed &= correct
            print(
                f"{hand:>5} {name:<6} asked {DEFAULT_FORCE:.1f} N on cell {CENTER_CELL + 1}: the pad read"
                f" {pad_force:.2f} N, most on {names[lit[0]]} cell {lit[1] + 1} (reading"
                f" {float(pressed_readings[lit]):.0f})  {'PASS' if correct else 'FAIL'}"
            )
    return passed


def main():
    viewport = KitVisualizerCfg(background_color=BACKGROUND)
    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYSICS_DT, render_interval=2, device=args_cli.device, visualizer_cfgs=viewport
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(TactileDemoSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    step = _hold_robot(sim, scene)
    step()  # the fingertip poses are read back from the simulation after a step
    print("[INFO]: Setup complete...")
    if args_cli.headless:
        passed = run_tour(scene, step)
        print("RESULT:", "PASS" if passed else "FAIL")
        # Closing the app can end the process, so the exit status has to go through it as well.
        status = 0 if passed else 1
        simulation_app.close(exit_code=status)
        sys.exit(status)
    run_remote(sim, scene, step)


def _hold_robot(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Puts the AI Worker in its demo pose and returns a function that advances the simulation by one step.

    Each step holds the base in place; the bench actuators hold every joint but the fingers.
    """
    robot = scene["robot"]
    # The AI Worker spawns in the pose stored in its USD file; put it in its configured pose right away.
    joint_pos, joint_vel = robot.data.default_joint_pos.torch.clone(), robot.data.default_joint_vel.torch.clone()
    thumb_ids, _ = robot.find_joints(list(OPEN_THUMBS), preserve_order=True)
    joint_pos[:, thumb_ids] = torch.tensor(list(OPEN_THUMBS.values()), device=sim.device)
    robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
    robot.actuators.target_command.set_position_index(value=joint_pos)
    robot.actuators.target_command.set_velocity_index(value=joint_vel)

    root_pose = robot.data.default_root_pose.torch.clone()
    root_pose[:, :3] += scene.env_origins
    still = torch.zeros(1, 6, device=sim.device)

    def step():
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(root_velocity=still)
        scene.write_data_to_sim()
        sim.step()
        scene.update(PHYSICS_DT)

    return step


def _hand_view(sensor: HandTactile) -> tuple[np.ndarray, np.ndarray]:
    """A camera eye and target that see all five fingertip pads of a hand.

    The thumb pad faces another way than the four others, so the camera looks along the direction halfway between.
    """
    poses = [sensor.cell_pose_w(finger, CENTER_CELL) for finger in range(len(sensor.finger_names))]
    thumb = sensor.finger_names.index("thumb")
    fingers_normal = np.mean([normal for finger, (_, normal) in enumerate(poses) if finger != thumb], axis=0)
    direction = fingers_normal / np.linalg.norm(fingers_normal) + poses[thumb][1]
    target = np.mean([point for point, _ in poses], axis=0)
    eye = target + 0.4 * direction / np.linalg.norm(direction) + np.array([0.0, 0.0, 0.03])
    return eye, target


def _place_heatmap(sensor: HandTactile):
    """Moves the hand's heatmap window into the top right corner of the viewport."""
    import omni.ui as ui

    heatmap, viewport = ui.Workspace.get_window(sensor.heatmap_title), ui.Workspace.get_window("Viewport")
    if heatmap and viewport:
        heatmap.position_x = viewport.position_x + viewport.width - heatmap.width - 12
        heatmap.position_y = viewport.position_y + 40


def _toward(position: np.ndarray, goal: np.ndarray, reach: float) -> np.ndarray:
    """``position`` moved at most ``reach`` towards ``goal``."""
    move = goal - position
    distance = np.linalg.norm(move)
    return goal.copy() if distance <= reach else position + move * (reach / distance)


def _pose(position: np.ndarray, device: str) -> torch.Tensor:
    """A root pose without rotation; quaternions are (x, y, z, w)."""
    return torch.tensor([[*position, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)


def _set(widget, attribute: str, value):
    """Sets a widget attribute only when it changes, so that unchanged widgets are not redrawn."""
    if getattr(widget, attribute) != value:
        setattr(widget, attribute, value)


if __name__ == "__main__":
    main()
    simulation_app.close()
