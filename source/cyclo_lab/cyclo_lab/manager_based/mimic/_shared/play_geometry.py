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

"""Render-only robot copies using geometry already loaded by Isaac Lab."""

from contextlib import nullcontext

import torch


class RobotGhost:
    """Display named link poses (xyzw) without creating a physics articulation."""

    def __init__(self, env, name, color):
        self.backends = []
        robot = env.scene['robot']
        visualizers = env.sim.visualizers
        if any(v.cfg.visualizer_type == 'kit' for v in visualizers):
            self.backends.append(KitRobotGhost(robot, name, color))
        if any(v.cfg.visualizer_type in ('newton_gl', 'viser') for v in visualizers):
            self.backends.append(make_newton_ghost(env.sim, robot, name, color))

    def is_visible(self):
        return all(backend.is_visible() for backend in self.backends)

    def set_visibility(self, visible):
        for backend in self.backends:
            backend.set_visibility(visible)

    def display(self, body_names, positions, orientations):
        for backend in self.backends:
            backend.display(body_names, positions, orientations)

    def close(self):
        for backend in self.backends:
            backend.close()


class KitRobotGhost:
    """Reference loaded visual prims, leaving the simulated robot untouched."""

    def __init__(self, robot, name, color):
        import isaaclab.sim as sim_utils
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

        self.stage = sim_utils.get_current_stage()
        self.path = f'/Visuals/TrackingPlayback/{name}'
        UsdGeom.Xform.Define(self.stage, self.path)
        self.links = {}
        root = sim_utils.find_matching_prims(robot.cfg.prim_path)[0]
        body_names = set(robot.body_names)
        cache = UsdGeom.XformCache()
        material = UsdShade.Material.Define(self.stage, self.path + '/material')
        shader = UsdShade.Shader.Define(self.stage, self.path + '/material/shader')
        from pxr import Sdf
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(0.65)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        # Snapshot traversal before adding references to the same stage.
        for source in list(Usd.PrimRange(root, Usd.TraverseInstanceProxies())):
            if not source.IsA(UsdGeom.Gprim):
                continue
            if source.HasAPI(UsdPhysics.CollisionAPI) or UsdGeom.Imageable(source).ComputeVisibility() == 'invisible':
                continue
            owner = source
            while owner and owner != root and (owner.GetName() not in body_names or not owner.HasAPI(UsdPhysics.RigidBodyAPI)):
                owner = owner.GetParent()
            if not owner or (owner.GetName() not in body_names or not owner.HasAPI(UsdPhysics.RigidBodyAPI)):
                continue
            body = owner.GetName()
            if body not in self.links:
                link = UsdGeom.Xform.Define(self.stage, f'{self.path}/{body}')
                self.links[body] = (link, link.AddTransformOp())
            destination = f'{self.path}/{body}/visual_{len(self.stage.GetPrimAtPath(self.path + "/" + body).GetChildren())}'
            prim = self.stage.DefinePrim(destination)
            prim.GetReferences().AddInternalReference(source.GetPath())
            prim.SetInstanceable(False)
            # Keep the original mesh scale/origin relative to its owning link.
            relative = cache.GetLocalToWorldTransform(source) * cache.GetLocalToWorldTransform(owner).GetInverse()
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(relative)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        self.root = UsdGeom.Imageable(self.stage.GetPrimAtPath(self.path))
        self.visible = True
        self.root.MakeInvisible()

    def is_visible(self):
        return self.visible

    def set_visibility(self, visible):
        self.visible = visible
        if visible:
            self.root.MakeVisible()
        else:
            self.root.MakeInvisible()

    def display(self, body_names, positions, orientations):
        from pxr import Gf

        if not self.visible:
            self.root.MakeInvisible()
            return
        self.root.MakeVisible()
        poses = dict(zip(body_names, zip(positions.detach().cpu().numpy(), orientations.detach().cpu().numpy()), strict=True))
        for name, (link, transform) in self.links.items():
            if name not in poses:
                link.MakeInvisible()
                continue
            link.MakeVisible()
            p, q = poses[name]
            matrix = Gf.Matrix4d(1)
            matrix.SetRotate(Gf.Quatd(float(q[3]), Gf.Vec3d(*map(float, q[:3]))))
            matrix.SetTranslateOnly(Gf.Vec3d(*map(float, p)))
            transform.Set(matrix)

    def close(self):
        self.stage.RemovePrim(self.path)


def make_newton_ghost(sim, robot, name, color):
    """Use Isaac Lab's marker render phase in both Newton GL and Viser.

    The marker subclass is registered through the existing registry, so calls
    occur inside the viewer's begin/end frame without patching either viewer.
    """
    from isaaclab_visualizers.newton.newton_visualization_markers import NewtonVisualizationMarkers
    from isaaclab.markers import VisualizationMarkersCfg
    import isaaclab.sim as sim_utils

    root = str(sim_utils.find_matching_prims(robot.cfg.prim_path)[0].GetPath())

    class LoadedRobotMarker(NewtonVisualizationMarkers):
        def __init__(self):
            super().__init__(VisualizationMarkersCfg(prim_path=f'/Visuals/TrackingPlayback/{name}', markers={}))
            self.poses = None
            self.models = {}

        def display(self, body_names, positions, orientations):
            self.poses = (tuple(body_names), positions.detach(), orientations.detach())

        def render(self, viewer, visible_env_ids=None, num_envs=1, render_id=None):
            import newton
            import warp as wp
            from newton.viewer import ViewerViser
            from isaaclab.utils.math import quat_apply, quat_mul

            if self.poses is None:
                return
            model = viewer.model
            names, positions, orientations = self.poses
            if id(model) not in self.models:
                bodies, flags = model.shape_body.numpy(), model.shape_flags.numpy()
                transforms = model.shape_transform.numpy()
                types, scales = model.shape_type.numpy(), model.shape_scale.numpy()
                parts = []
                for i, body in enumerate(bodies):
                    if body < 0 or not (flags[i] & int(newton.ShapeFlags.VISIBLE)):
                        continue
                    label = model.body_label[body]
                    if not label.startswith(root + '/'):
                        continue
                    parts.append((i, label.rsplit('/', 1)[-1], int(types[i]), scales[i],
                                  model.shape_source[i], positions.new_tensor(transforms[i])))
                self.models[id(model)] = dict(
                    parts=parts,
                    colors=wp.array([color], dtype=wp.vec3, device=viewer.device),
                    opacities=wp.array([0.65], dtype=wp.float32, device=viewer.device),
                    identity=positions.new_tensor([[0, 0, 0, 0, 0, 0, 1]]),
                )
            cached = self.models[id(model)]
            # Viser's begin/end_frame do not batch messages. Group every body
            # update so the browser does not display a partially updated ghost.
            with viewer._server.atomic() if isinstance(viewer, ViewerViser) else nullcontext():
                for index, body, kind, scale, source, local in cached['parts']:
                    path = f'{self.group_id}/shape_{index}'
                    visible = self.visible and body in names and (visible_env_ids is None or 0 in visible_env_ids)
                    if body in names:
                        i = names.index(body)
                        p = positions[i] + quat_apply(orientations[i], local[:3])
                        q = quat_mul(orientations[i], local[3:])
                        pose = torch.cat((p, q)).reshape(1, 7).contiguous()
                    else:
                        pose = cached['identity']
                    viewer.log_shapes(
                        path, kind, scale, wp.from_torch(pose, dtype=wp.transform),
                        geo_src=source, colors=cached['colors'], opacities=cached['opacities'], hidden=not visible,
                    )

        def close(self):
            self.set_visibility(False)
            # Publish hidden instances before removing the registry entry.
            for visualizer in sim.visualizers:
                if visualizer.cfg.visualizer_type in ('newton_gl', 'viser'):
                    sim.render()
                    break
            super().close()

    return LoadedRobotMarker()
