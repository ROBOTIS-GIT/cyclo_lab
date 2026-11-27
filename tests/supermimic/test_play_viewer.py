from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from cyclo_lab.manager_based.mimic._shared.play_geometry import KitRobotGhost, make_newton_ghost
from cyclo_lab.manager_based.mimic._shared.play_viewer import TrackingPlayViewer


def test_kit_reuses_loaded_visual_and_preserves_local_transform(monkeypatch):
    import isaaclab.sim as sim_utils
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    body = UsdGeom.Xform.Define(stage, "/World/Robot/arm")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    body.AddTranslateOp().Set(Gf.Vec3d(10, 0, 0))
    visual = UsdGeom.Cube.Define(stage, "/World/Robot/arm/visual")
    visual.CreateSizeAttr(2)
    visual.AddTranslateOp().Set(Gf.Vec3d(0, 2, 0))
    collision = UsdGeom.Sphere.Define(stage, "/World/Robot/arm/collision")
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    monkeypatch.setattr(sim_utils, "get_current_stage", lambda: stage)
    monkeypatch.setattr(sim_utils, "find_matching_prims", lambda _: [root])
    before = stage.GetRootLayer().ExportToString()
    ghost = KitRobotGhost(NS(cfg=NS(prim_path=str(root.GetPath())), body_names=["arm"]), "test", (0, 1, 0))
    ghost.display(["arm"], torch.tensor([[1., 0, 0]]), torch.tensor([[0., 0, 0, 1.]]))
    copied = stage.GetPrimAtPath('/Visuals/TrackingPlayback/test/arm/visual_0')
    assert copied.HasAuthoredReferences()
    assert UsdGeom.Cube(copied).GetSizeAttr().Get() == 2
    np.testing.assert_allclose(UsdGeom.XformCache().GetLocalToWorldTransform(copied).ExtractTranslation(), [1, 2, 0])
    ghost.display(["arm"], torch.tensor([[1., 0, 0]]),
                  torch.tensor([[0., 0, 2**-0.5, 2**-0.5]]))
    np.testing.assert_allclose(UsdGeom.XformCache().GetLocalToWorldTransform(copied).ExtractTranslation(),
                               [-1, 0, 0], atol=1e-6)
    assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI)
                   for p in Usd.PrimRange(stage.GetPrimAtPath('/Visuals/TrackingPlayback/test')))
    ghost.display([], torch.empty(0, 3), torch.empty(0, 4))
    assert UsdGeom.Imageable(stage.GetPrimAtPath('/Visuals/TrackingPlayback/test/arm')).ComputeVisibility() == 'invisible'
    ghost.close()
    # Only empty parent containers may remain; the robot was not edited.
    stage.RemovePrim('/Visuals')
    assert stage.GetRootLayer().ExportToString() == before


def test_newton_marker_reuses_model_shapes_and_link_poses(monkeypatch):
    import isaaclab.sim as sim_utils
    import newton
    from isaaclab.markers.vis_marker_registry import VisMarkerRegistry

    monkeypatch.setattr(sim_utils, 'find_matching_prims', lambda _: [NS(GetPath=lambda: '/World/Robot')])
    sim = NS(vis_marker_registry=VisMarkerRegistry(), visualizers=[])
    monkeypatch.setattr(sim_utils.SimulationContext, 'instance', lambda: sim)
    robot = NS(cfg=NS(prim_path='/World/Robot'))
    source = newton.Mesh(np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=np.float32), np.array([0,1,2]))
    model = NS(shape_body=NS(numpy=lambda: np.array([0, 1, 0])),
               shape_flags=NS(numpy=lambda: np.array([int(newton.ShapeFlags.VISIBLE)] * 2 + [0])),
               shape_transform=NS(numpy=lambda: np.array([[0,2,0,0,0,0,1]] * 3)),
               shape_type=NS(numpy=lambda: np.array([int(newton.GeoType.MESH)] * 3)),
               shape_scale=NS(numpy=lambda: np.ones((3,3))), shape_source=[source] * 3,
               body_label=['/World/Robot/arm', '/World/envs/env_1/Robot/arm'])
    calls = []
    viewer = NS(model=model, device='cpu', log_shapes=lambda *a, **kw: calls.append((a,kw)))
    ghost = make_newton_ghost(sim, robot, 'test', (0,1,0))
    assert ghost.cfg.prim_path == '/Visuals/TrackingPlayback/test'
    assert sim.vis_marker_registry.get_groups()[ghost.group_id] is ghost
    ghost.display(['arm'], torch.tensor([[3.,0,0]]), torch.tensor([[0.,0,0,1.]]))
    ghost.render(viewer)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs['geo_src'] is source and not kwargs['hidden']
    np.testing.assert_allclose(args[3].numpy()[0], [3,2,0,0,0,0,1])
    # Static shape transforms are uploaded once, not reread on each frame.
    model.shape_transform.numpy = lambda: pytest.fail('Shape transforms must be cached')
    ghost.display(['arm'], torch.tensor([[3.,0,0]]), torch.tensor([[0.,0,2**-0.5,2**-0.5]]))
    ghost.render(viewer)
    np.testing.assert_allclose(calls[-1][0][3].numpy()[0, :3], [1,0,0], atol=1e-6)
    assert calls[-1][1]['colors'] is kwargs['colors']
    assert calls[-1][1]['opacities'] is kwargs['opacities']
    # The official Newton panel uses this method, independently of our controls.
    ghost.set_visibility(False)
    ghost.display(['arm'], torch.tensor([[3.,0,0]]), torch.tensor([[0.,0,0,1.]]))
    ghost.render(viewer)
    assert calls[-1][1]['hidden'] and not ghost.is_visible()
    ghost.set_visibility(True)
    ghost.render(viewer)
    assert not calls[-1][1]['hidden']
    ghost.display([], torch.empty(0,3), torch.empty(0,4))
    ghost.render(viewer)
    assert calls[-1][1]['hidden']
    ghost.close()
    assert not sim.vis_marker_registry.get_groups()


def controller():
    viewer = object.__new__(TrackingPlayViewer)
    calls = []
    viewer.command = NS(frame_ids=torch.tensor([0, 0]), playback_frames=None)
    viewer.runner = NS(env=NS(reset=lambda: calls.append("reset")),
                       reset_playback_history=lambda: calls.append("history"))
    viewer.predictor = NS(reset=lambda: calls.append("prediction"))
    viewer.prediction = None
    backend = NS(visible=True)
    backend.is_visible = lambda: backend.visible
    backend.set_visibility = lambda visible: setattr(backend, 'visible', visible)
    from cyclo_lab.manager_based.mimic._shared.play_geometry import RobotGhost
    viewer.reference = object.__new__(RobotGhost)
    viewer.reference.backends = [backend]
    viewer.starts = [0, 80, 130]
    viewer.requested_clip = None
    viewer.clip_index = 0
    return viewer, calls


def test_select_and_restart_use_clip_start_and_reset_history():
    viewer, calls = controller()
    for _ in range(2):
        viewer.select_clip(1)
        viewer.before_step()
        torch.testing.assert_close(viewer.command.playback_frames, torch.tensor([80, 80]))
        assert viewer.requested_clip is None
    assert calls == ["reset", "history", "prediction"] * 2


def test_newton_panel_selects_clip_without_kit():
    viewer, calls = controller()
    viewer.closed = False
    viewer.clip_names = ["dance1", "dance2", "dance3"]
    viewer.velocity_ranges = None
    viewer.show_ref = viewer.show_pred = viewer.separate = True
    imgui = NS(Cond_=NS(once=1), set_next_item_open=lambda *args: None,
               collapsing_header=lambda _: True, combo=lambda *args: (True, 2),
               button=lambda _: False, checkbox=lambda label, value: (False, value))
    viewer._draw_newton_controls(imgui)
    viewer.before_step()
    torch.testing.assert_close(viewer.command.playback_frames, torch.tensor([130, 130]))
    assert calls == ["reset", "history", "prediction"]


def test_viser_controls_select_restart_and_toggle():
    import viser

    viewer, calls = controller()
    viewer.clip_names = ["motion", "motion", "motion"]
    viewer.velocity_ranges = None
    viewer.viser_controls = []
    viewer.show_ref = viewer.separate = True
    server = viser.ViserServer(host="127.0.0.1", port=0)
    try:
        viewer._create_viser_controls(server)
        _, clip, restart, reference, separate = viewer.viser_controls
        clip.value = "2. motion"
        for callback in clip._impl.update_cb:
            callback(NS(target=clip))
        viewer.before_step()
        torch.testing.assert_close(viewer.command.playback_frames, torch.tensor([80, 80]))
        for callback in restart._impl.update_cb:
            callback(NS(target=restart))
        viewer.before_step()
        assert calls == ["reset", "history", "prediction"] * 2
        reference.value = False
        for callback in reference._impl.update_cb:
            callback(NS(target=reference))
        assert not viewer.show_ref
        assert separate.value
    finally:
        for handle in reversed(viewer.viser_controls):
            handle.remove()
        server.stop()


@pytest.mark.parametrize("backend", [None, "kit", "newton_gl", "viser"])
def test_viewer_creation_uses_active_visualizer(monkeypatch, backend):
    visualizers = [NS(cfg=NS(visualizer_type=backend))] if backend else []
    runner = NS(bridge=NS(env=NS(sim=NS(visualizers=visualizers))))
    monkeypatch.setattr(TrackingPlayViewer, "__init__", lambda self, *args, **kwargs: None)
    assert (TrackingPlayViewer.create(runner, []) is not None) == bool(backend)


def test_hiding_kit_controls_does_not_stop_playback():
    viewer, _ = controller()
    viewer.closed = False
    viewer.window = NS(visible=False)
    kit = NS(running=True)
    kit.is_running = lambda: kit.running
    viewer.visualizers = [kit]
    assert viewer.is_running()
    kit.running = False
    assert not viewer.is_running()


def test_tracking_controls_read_backend_visibility():
    viewer, _ = controller()
    backend = viewer.reference.backends[0]
    backend.set_visibility(False)
    assert not viewer.show_ref
    viewer.show_ref = True
    assert backend.is_visible()


def test_explicit_playback_frames_take_precedence_over_sampling():
    from cyclo_lab.manager_based.mimic._shared.motion_library_command import MotionLibraryCommand

    command = object.__new__(MotionLibraryCommand)
    command.cfg = NS(start_from_zero=True, clip_timeout=False)
    command.frame_ids = torch.tensor([2, 4])
    command.motion_ends = torch.tensor([3, 8])
    command.playback_frames = torch.tensor([1, 3])
    command._sample_start_frames(torch.tensor([1]))
    assert command.frame_ids.tolist() == [2, 3]
    command.playback_frames = None
    command._sample_start_frames(torch.tensor([1]))
    assert command.frame_ids.tolist() == [2, 3]


def test_motion_sampler_resumes_after_clearing_playback_selection():
    from cyclo_lab.manager_based.mimic._shared.motion_library_command import MotionLibraryCommand

    command = object.__new__(MotionLibraryCommand)
    command.cfg = NS(start_from_zero=False)
    command.frame_ids = torch.tensor([2, 4])
    command.playback_frames = torch.tensor([1, 3])
    calls = []

    def sample(ids):
        calls.append(ids.tolist())
        return torch.full_like(ids, 5)

    command.sampler = NS(sample=sample)
    command._sample_start_frames(torch.tensor([1]))
    assert not calls
    assert command.frame_ids.tolist() == [2, 3]
    command.playback_frames = None
    command._sample_start_frames(torch.tensor([1]))
    assert calls == [[1]]
    assert command.frame_ids.tolist() == [2, 5]
