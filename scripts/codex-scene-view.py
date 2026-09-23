"""Presentation settings for the service's dedicated live scene-only client."""


def configure_scene_view(env_cfg, visualizer_factory=None):
    """Disable debug overlays/plot collection before creating the environment.

    This preserves sensor computations, reward terms, and camera interaction.
    Existing camera and backend settings are retained.
    """
    sim = env_cfg.sim
    if sim.default_visualizer_cfg is None:
        if visualizer_factory is None:
            from isaaclab.visualizers import VisualizerCfg

            visualizer_factory = VisualizerCfg
        sim.default_visualizer_cfg = visualizer_factory()
    explicit = sim.visualizer_cfgs
    if not isinstance(explicit, (list, tuple)):
        explicit = [explicit]
    for cfg in [sim.default_visualizer_cfg, *explicit]:
        if cfg is not None:
            cfg.enable_markers = False
            cfg.enable_live_plots = False
    # Disable marker producers while retaining the commands and sensor objects.
    for section in (getattr(env_cfg, "commands", None), getattr(env_cfg, "scene", None)):
        if section is not None:
            for value in vars(section).values():
                if hasattr(value, "debug_vis"):
                    value.debug_vis = False


def clear_viewer_controls(server):
    """Remove authored simulator controls; the custom client omits its UI shell.

    This is deliberately not a replacement for the scene-only client build:
    upstream Viser still renders its built-in controls after gui.reset().
    """
    server.gui.reset()
    server.gui.configure_theme(show_logo=False, show_share_button=False)
