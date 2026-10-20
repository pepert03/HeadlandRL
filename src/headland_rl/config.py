"""Construccion del entorno a partir de una config YAML de entrenamiento."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import yaml

from .envs.headland_env import HeadlandTurnEnv
from .envs.params import CropParams, VehicleParams
from .envs.wrappers import NormalizedActions


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_env(cfg: dict, seed: int | None = None, **overrides) -> gym.Env:
    """Entorno identico al de entrenamiento; `overrides` pisa argumentos del env.

    Con `normalize_actions: true` en la config se envuelve en NormalizedActions
    (la politica actua en [-1, 1]); el entorno base queda en `env.unwrapped`.
    """
    ec = cfg.get("env", {})
    ref = cfg.get("reference", None)
    kwargs = dict(
        dr_mode=cfg.get("dr_mode", "fixed"),
        privileged=cfg.get("privileged", False),
        dt=ec.get("dt", 0.05),
        max_steps=ec.get("max_steps", 1000),
        pos_tol=ec.get("pos_tol", 0.30),
        ang_tol_deg=ec.get("ang_tol_deg", 10.0),
        fixed_vehicle=VehicleParams(**ec["fixed_vehicle"]) if ec.get("fixed_vehicle") else None,
        fixed_crop=CropParams(**ec["fixed_crop"]) if ec.get("fixed_crop") else None,
        reference_poses=np.load(ref) if ref else None,
        **ec.get("reward", {}),
    )
    kwargs.update(overrides)
    env = HeadlandTurnEnv(**kwargs)
    if cfg.get("normalize_actions", False):
        env = NormalizedActions(env)
    return env
