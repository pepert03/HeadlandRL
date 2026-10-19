"""HeadlandRL: entorno parametrizable de cambio de calle para RL."""
from .envs.headland_env import HeadlandTurnEnv
from .envs.params import CropParams, DomainRanges, EpisodeParams, VehicleParams
from .envs.randomization import DRMode, EvalRegime

__all__ = [
    "HeadlandTurnEnv",
    "VehicleParams",
    "CropParams",
    "DomainRanges",
    "EpisodeParams",
    "DRMode",
    "EvalRegime",
]

__version__ = "0.1.0"

try:
    import gymnasium as gym

    gym.register(
        id="HeadlandTurn-v0",
        entry_point="headland_rl.envs.headland_env:HeadlandTurnEnv",
    )
except Exception:
    pass
