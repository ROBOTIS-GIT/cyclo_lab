"""Kit controls and render-only tracked-link overlays for the first playback environment."""

from pathlib import Path

import torch


class TrackedLinks:
    """Four limb chains and the spine in the shared K1 tracking body order."""

    CHAINS = ((0, 1, 2, 3), (0, 4, 5, 6), (0, 7), (7, 8, 9, 10), (7, 11, 12, 13))

    def __init__(self, stage, name, color):
        from pxr import UsdGeom

        self.curves = UsdGeom.BasisCurves.Define(stage, f"/Visuals/TrackingPlayback/{name}")
        self.curves.CreateTypeAttr("linear")
        self.curves.CreateCurveVertexCountsAttr([len(chain) for chain in self.CHAINS])
        self.curves.CreateWidthsAttr([0.045])
        self.curves.SetWidthsInterpolation("constant")
        self.curves.CreateDisplayColorAttr([color])
        self.indices = [i for chain in self.CHAINS for i in chain]

    def display(self, positions, offset, visible):
        if visible:
            self.curves.MakeVisible()
            self.curves.GetPointsAttr().Set((positions[self.indices] + offset).detach().cpu().numpy())
        else:
            self.curves.MakeInvisible()


class TrackingPlayViewer:
    def __init__(self, runner, motion_files, *, predictor=None, extra_controls=None):
        import omni.ui as ui
        import omni.usd

        self.runner = runner
        self.command = runner.bridge.env.command_manager.get_term("reference_trajectory")
        self.requested_clip = None
        self.starts = [0, *self.command.motion_ends.tolist()[:-1]]
        stage = omni.usd.get_context().get_stage()
        self.reference = TrackedLinks(stage, "Reference", (0.15, 0.9, 0.25))
        self.predictor = predictor
        self.prediction = TrackedLinks(stage, "Prediction", (0.08, 0.4, 1.0)) if predictor else None
        self.window = ui.Window("Motion tracking", width=330, height=275 if extra_controls else 180)
        with self.window.frame:
            with ui.VStack(height=0, spacing=4):
                with ui.HStack():
                    ui.Label("Motion")
                    self.clip = ui.ComboBox(0, *(Path(path).stem for path in motion_files))
                self.clip.model.add_item_changed_fn(lambda model, _: self.select_clip())
                ui.Button("Restart selected motion", clicked_fn=self.select_clip)
                if extra_controls:
                    extra_controls(ui)
                with ui.CollapsableFrame("Debug visualization"):
                    with ui.VStack():
                        self.show_ref = self.checkbox(ui, "Reference motion (green)")
                        if self.prediction:
                            self.show_pred = self.checkbox(ui, "World model rollout (blue)")
                        self.separate = self.checkbox(ui, "Separate overlays")

    @staticmethod
    def checkbox(ui, label):
        with ui.HStack():
            toggle = ui.CheckBox(width=22)
            toggle.model.set_value(True)
            ui.Label(label)
        return toggle.model

    def select_clip(self):
        self.requested_clip = self.clip.model.get_item_value_model().as_int

    def before_step(self):
        if self.requested_clip is not None:
            r = self.runner
            self.command.playback_frames = torch.full_like(
                self.command.frame_ids, self.starts[self.requested_clip]
            )
            try:
                r.env.reset()
            finally:
                self.command.playback_frames = None
            r.reset_playback_history()
            if self.predictor:
                self.predictor.reset()
            self.requested_clip = None
        if self.prediction:
            self.predicted = self.predictor.advance()

    def draw(self):
        bridge = self.runner.bridge
        frame = bridge.frame_ids()[:1]
        reference = bridge.motion.body_state(frame)[0][0]
        origin = bridge.env.scene.env_origins[0]
        offset = origin.new_tensor((0.0, 0.9 if self.separate.as_bool else 0.0, 0.0))
        self.reference.display(reference, origin - offset, self.show_ref.as_bool)
        if self.prediction:
            self.prediction.display(self.predicted, origin + offset, self.show_pred.as_bool)

    def close(self):
        self.window.destroy()
        self.reference.curves.MakeInvisible()
        if self.prediction:
            self.prediction.curves.MakeInvisible()
