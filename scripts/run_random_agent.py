"""Smoke test: agente aleatorio sobre el entorno (sin entrenamiento)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import yaml

from headland_rl.envs.headland_env import HeadlandTurnEnv
from headland_rl.envs.params import CropParams, DomainRanges, InitNoise, VehicleParams


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env_default.yaml")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--dr-mode", default=None)
    ap.add_argument("--privileged", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    env = HeadlandTurnEnv(
        dr_mode=args.dr_mode or cfg.get("dr_mode", "joint"),
        privileged=args.privileged or cfg.get("privileged", False),
        dt=cfg.get("dt", 0.05),
        max_steps=cfg.get("max_steps", 400),
        fixed_vehicle=VehicleParams(**cfg.get("fixed_vehicle", {})) if cfg.get("fixed_vehicle") else None,
        fixed_crop=CropParams(**cfg.get("fixed_crop", {})) if cfg.get("fixed_crop") else None,
        init_noise=InitNoise(**cfg.get("init_noise", {})) if cfg.get("init_noise") else None,
    )
    rng = np.random.default_rng(args.seed)
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        ret, steps = 0.0, 0
        while not done:
            act = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(act)
            ret += rew
            steps += 1
            done = term or trunc
        print(
            f"ep={ep} steps={steps} ret={ret:.2f} success={info['success']} "
            f"collided={info['collided']}({info['collision_reason']}) "
            f"dpos={info['dpos']:.2f}m dtheta={info['dtheta_deg']:.1f}deg "
            f"Rmin={info['R_min']:.2f}m H={env.get_params().crop.headland_width:.2f}m"
        )
    img = env.render()
    Path("runs").mkdir(exist_ok=True)
    import matplotlib.pyplot as plt

    plt.imsave("runs/smoke_last_frame.png", img)
    print("Frame guardado en runs/smoke_last_frame.png, obs_dim:", env.observation_space.shape)


if __name__ == "__main__":
    main()
