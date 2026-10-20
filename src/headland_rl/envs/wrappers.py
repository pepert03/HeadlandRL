"""Wrappers del entorno para el entrenamiento con SB3."""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .headland_env import DEFAULT_DELTA_ABS_DEG


class NormalizedActions(gym.ActionWrapper):
    """Acciones de la politica en [-1, 1]^2 -> ordenes fisicas del entorno.

    v:     a0 >= 0 -> a0 * v_max ; a0 < 0 -> a0 * v_rev_max (0 sigue siendo parar)
    delta: a1 * 35 deg (el entorno lo recorta al delta_max del episodio)
    El entorno base sigue aceptando unidades fisicas (controladores clasicos).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        veh = env.unwrapped.fixed_vehicle
        self.v_max, self.v_rev = veh.v_max, veh.v_rev_max
        self.d_abs = math.radians(DEFAULT_DELTA_ABS_DEG)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        v = a[0] * (self.v_max if a[0] >= 0 else self.v_rev)
        return np.array([v, a[1] * self.d_abs])
