"""Visor en vivo del entorno (ventana matplotlib interactiva)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
try:
    matplotlib.use("TkAgg")  # backend interactivo (ventana)
except Exception:
    pass
import matplotlib.pyplot as plt
import numpy as np

from headland_rl.envs.headland_env import HeadlandTurnEnv


def main(episodes: int = 3, seed: int = 0, pause: float = 0.02):
    env = HeadlandTurnEnv(dr_mode="joint")
    fig, ax_img = plt.subplots(figsize=(7, 5))
    ax_img.axis("off")
    img_handle = None

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done and plt.fignum_exists(fig.number):
            act = env.action_space.sample()
            _, _, term, trunc, info = env.step(act)
            done = term or trunc
            frame = env.render()
            if img_handle is None:
                img_handle = ax_img.imshow(frame)
            else:
                img_handle.set_data(frame)
            ax_img.set_title(
                f"ep={ep} step={info['steps']} dpos={info['dpos']:.2f}m "
                f"{'OK' if info['success'] else info['collision_reason'] if info['collided'] else ''}"
            )
            plt.pause(pause)
        print(
            f"ep={ep} success={info['success']} collided={info['collided']} "
            f"({info['collision_reason']}) dpos={info['dpos']:.2f}m"
        )
    print("Cierra la ventana para terminar.")
    plt.show()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pause", type=float, default=0.02)
    args = ap.parse_args()
    main(args.episodes, args.seed, args.pause)
