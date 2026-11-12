"""FFW-SH5 bringup environment registrations."""

import gymnasium as gym

_ENTRY_POINT = "cyclo_lab.manager_based.continuous_env:ContinuousManagerBasedEnv"

gym.register(
    id="Cyclo-Bringup-FFW-SH5-v0",
    entry_point=_ENTRY_POINT,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FFWSH5BringupEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Cyclo-Bringup-Simple-Warehouse-FFW-SH5-v0",
    entry_point=_ENTRY_POINT,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FFWSH5SimpleWarehouseBringupEnvCfg",
    },
    disable_env_checker=True,
)
