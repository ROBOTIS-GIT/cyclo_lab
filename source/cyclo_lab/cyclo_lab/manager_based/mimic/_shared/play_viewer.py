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
# Author: Kiwoong Park

"""Motion selection, restart and reference rendering in Kit, Newton GL and Viser."""

from pathlib import Path

import torch

from .play_geometry import RobotGhost


class TrackingPlayViewer:
    @classmethod
    def create(cls, runner, motion_files, *, predictor=None, velocity_ranges=None):
        sim = runner.bridge.env.sim
        newton = [v for v in sim.visualizers if v.cfg.visualizer_type == "newton_gl"]
        viser = [v for v in sim.visualizers if v.cfg.visualizer_type == "viser"]
        kit = any(v.cfg.visualizer_type == "kit" for v in sim.visualizers)
        if not kit and not newton and not viser:
            return None
        return cls(runner, motion_files, kit=kit, newton=newton, viser=viser,
                   predictor=predictor, velocity_ranges=velocity_ranges)

    def __init__(self, runner, motion_files, *, kit, newton, viser=(), predictor=None, velocity_ranges=None):
        self.runner = runner
        self.visualizers = runner.bridge.env.sim.visualizers
        self.command = runner.bridge.env.command_manager.get_term("reference_trajectory")
        self.requested_clip = 0
        self.clip_index = 0
        self.clip_names = [Path(path).stem for path in motion_files]
        self.starts = [0, *self.command.motion_ends.tolist()[:-1]]
        self.predictor = predictor
        self.velocity_ranges = velocity_ranges
        self.viser_controls = []
        self.window = None
        self.closed = False
        self.separate = True
        bridge = runner.bridge
        self.reference = RobotGhost(bridge.env, "Reference", (0.15, 0.9, 0.25))
        self.prediction = None
        if predictor:
            self.prediction = RobotGhost(bridge.env, "Prediction", (0.08, 0.4, 1.0))
        if kit:
            self._create_kit_controls()
        for visualizer in newton:
            visualizer.register_ui_callback(self._draw_newton_controls, position="panel")
        for visualizer in viser:
            # Isaac Lab exposes Viser controls through its underlying viewer.
            self._create_viser_controls(visualizer._viewer._server)

    @property
    def show_ref(self):
        return self.reference.is_visible()

    @show_ref.setter
    def show_ref(self, visible):
        self.reference.set_visibility(visible)

    @property
    def show_pred(self):
        return bool(self.prediction and self.prediction.is_visible())

    @show_pred.setter
    def show_pred(self, visible):
        if self.prediction:
            self.prediction.set_visibility(visible)

    def _create_viser_controls(self, server):
        folder = server.gui.add_folder("Motion tracking")
        self.viser_controls.append(folder)
        with folder:
            # Viser selects by label rather than index; keep duplicate file stems distinct.
            options = tuple(f"{index + 1}. {name}" for index, name in enumerate(self.clip_names))
            clip = server.gui.add_dropdown("Motion", options=options, initial_value=options[0])
            clip.on_update(lambda event: self.select_clip(options.index(event.target.value)))
            restart = server.gui.add_button("Restart selected motion")
            restart.on_click(lambda _: self.select_clip(self.clip_index))
            self.viser_controls.extend([clip, restart])
            if self.velocity_ranges:
                for axis, (label, bounds) in enumerate(self.velocity_ranges):
                    slider = server.gui.add_slider(label, min=bounds[0], max=bounds[1], step=0.01,
                                                   initial_value=float(self.runner.play_velocity[axis]))
                    slider.on_update(lambda event, axis=axis: self.set_velocity(axis, event.target.value))
                    self.viser_controls.append(slider)
            toggles = [("Reference robot (green)", "show_ref"), ("Separate overlays", "separate")]
            if self.prediction:
                toggles.append(("World model rollout (blue)", "show_pred"))
            for label, attribute in toggles:
                toggle = server.gui.add_checkbox(label, initial_value=getattr(self, attribute))
                toggle.on_update(lambda event, attribute=attribute: setattr(self, attribute, event.target.value))
                self.viser_controls.append(toggle)

    def _create_kit_controls(self):
        import omni.ui as ui

        self.window = ui.Window("Motion tracking", width=330, height=290 if self.velocity_ranges else 200)
        self.window.position_x = 20
        self.window.position_y = 60
        with self.window.frame:
            with ui.VStack(height=0, spacing=4):
                with ui.HStack():
                    ui.Label("Motion")
                    self.clip = ui.ComboBox(0, *self.clip_names)
                self.clip.model.add_item_changed_fn(
                    lambda model, _: self.select_clip(model.get_item_value_model().as_int)
                )
                ui.Button("Restart selected motion", clicked_fn=lambda: self.select_clip(self.clip_index))
                if self.velocity_ranges:
                    for axis, (label, bounds) in enumerate(self.velocity_ranges):
                        with ui.HStack():
                            ui.Label(label)
                            slider = ui.FloatSlider(min=bounds[0], max=bounds[1])
                            slider.model.set_value(float(self.runner.play_velocity[axis]))
                            slider.model.add_value_changed_fn(
                                lambda model, axis=axis: self.set_velocity(axis, model.as_float)
                            )
                with ui.CollapsableFrame("Debug visualization"):
                    with ui.VStack():
                        self._kit_checkbox(ui, "Reference robot (green)", "show_ref")
                        if self.prediction:
                            self._kit_checkbox(ui, "World model rollout (blue)", "show_pred")
                        self._kit_checkbox(ui, "Separate overlays", "separate")

    def _kit_checkbox(self, ui, label, attribute):
        with ui.HStack():
            toggle = ui.CheckBox(width=22)
            toggle.model.set_value(getattr(self, attribute))
            toggle.model.add_value_changed_fn(lambda model: setattr(self, attribute, model.as_bool))
            ui.Label(label)

    def _draw_newton_controls(self, imgui):
        if self.closed:
            return
        imgui.set_next_item_open(True, imgui.Cond_.once)
        if not imgui.collapsing_header("Motion tracking"):
            return
        changed, selected = imgui.combo("Motion", self.clip_index, self.clip_names)
        if changed:
            self.select_clip(selected)
        if imgui.button("Restart selected motion"):
            self.select_clip(self.clip_index)
        if self.velocity_ranges:
            for axis, (label, bounds) in enumerate(self.velocity_ranges):
                changed, value = imgui.slider_float(label, float(self.runner.play_velocity[axis]), *bounds)
                if changed:
                    self.set_velocity(axis, value)
        _, self.show_ref = imgui.checkbox("Reference robot (green)", self.show_ref)
        if self.prediction:
            _, self.show_pred = imgui.checkbox("World model rollout (blue)", self.show_pred)
        _, self.separate = imgui.checkbox("Separate overlays", self.separate)

    def set_velocity(self, axis, value):
        self.runner.play_velocity[axis] = value

    def select_clip(self, index):
        self.clip_index = index
        self.requested_clip = index

    def is_running(self):
        if self.closed:
            return False
        return any(v.is_running() for v in self.visualizers)

    def before_step(self):
        if self.requested_clip is not None:
            r = self.runner
            self.command.playback_frames = torch.full_like(self.command.frame_ids, self.starts[self.requested_clip])
            r.env.reset()
            r.reset_playback_history()
            if self.predictor:
                self.predictor.reset()
            self.requested_clip = None
        if self.prediction:
            self.predicted = self.predictor.advance()

    def draw(self):
        bridge = self.runner.bridge
        frame = bridge.frame_ids()[:1]
        origin = bridge.env.scene.env_origins[0]
        offset = origin.new_tensor((0.0, 0.9 if self.separate else 0.0, 0.0))
        reference = self.command.reference
        self.reference.display(
            bridge.robot.body_names,
            reference._body_position_w[frame[0]] + origin - offset,
            reference._body_orientation_w[frame[0]],
        )
        if self.prediction:
            names, positions, orientations = self.predicted
            self.prediction.display(names, positions + origin + offset, orientations)

    def close(self):
        self.closed = True
        self.command.playback_frames = None
        if self.window:
            self.window.destroy()
        for handle in reversed(self.viser_controls):
            handle.remove()
        self.reference.close()
        if self.prediction:
            self.prediction.close()
