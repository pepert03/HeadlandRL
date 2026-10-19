"""Sonda de backplay: tasa de exito desde cada fraccion de arco (diagnostico)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from stable_baselines3 import PPO

from headland_rl.envs.headland_env import HeadlandTurnEnv

model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/A/seed_0/ppo_final.zip"
m = PPO.load(model_path)
for u in (1.0, 0.9, 0.7, 0.5, 0.3, 0.15, 0.0):
    env = HeadlandTurnEnv(dr_mode="fixed", max_steps=500)
    env._curr_on = True
    env._u_min = u
    env._u_max = u  # punto (sin mezcla): diagnostico por tramo
    env._v_cap = None
    ok, lens = 0, []
    for ep in range(5):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            a, _ = m.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        ok += int(info["success"])
        lens.append(info["steps"])
    print(f"u={u:.2f}: success={ok}/5 len_mean={np.mean(lens):.0f}")
