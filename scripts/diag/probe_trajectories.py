"""Sonda: dibuja trayectorias estocasticas del modelo actual (diagnostico)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO

from headland_rl.envs.headland_env import HeadlandTurnEnv

model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/A/seed_0/ppo_final.zip"
m = PPO.load(model_path)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, ax in enumerate(axes):
    env = HeadlandTurnEnv(dr_mode="fixed", max_steps=500)
    obs, _ = env.reset(seed=100 + i)
    done = False
    while not done:
        a, _ = m.predict(obs, deterministic=False)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    tr = np.array(env._trail)
    ax.plot(tr[:, 0], tr[:, 1], "b-", lw=1)
    ax.plot([0], [0], "go")
    ax.plot([2.5], [0], "r*", ms=10)
    ax.add_patch(plt.Rectangle((-2, 0), 6.5, 6.0, color="#f5f0dd", alpha=0.5))
    ax.set_aspect("equal")
    ax.set_title(f"ep{i}: dpos={info['dpos']:.2f} coll={info['collided']}")
fig.tight_layout()
fig.savefig("runs/traj_probe.png")
print("saved runs/traj_probe.png")
