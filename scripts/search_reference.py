"""Busqueda aleatoria de una maniobra en 4 segmentos (referencia open-loop).

El entorno es determinista: basta UNA secuencia que complete la maniobra en
estricto para usarla como referencia de backplay (+ demos de BC).
Espacio: por segmento (v, delta_deg, duracion). Rapido (~miles de rollouts/s).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math
import numpy as np

from headland_rl.envs.geometry import goal_pose, wrap_angle
from headland_rl.envs.headland_env import HeadlandTurnEnv

N_SEG = 3
N_TRIALS = 4000
VS = [-0.8, 0.8]
DELTAS = [-26.0, -13.0, 0.0, 13.0, 26.0]
DURS = (15, 90)


def rollout_open(env, seq, seed=0, hist=None):
    """Ejecuta segmentos open-loop. Retorna (info, termino_antes)."""
    env.reset(seed=seed)
    for v, dd, n in seq:
        for _ in range(n):
            obs = env._get_obs().copy()
            _, _, term, trunc, info = env.step([v, math.radians(dd)])
            if hist is not None:
                hist.append((obs, np.array([v, math.radians(dd)], dtype=np.float32)))
            if term or trunc:
                return info, True
    return info, False


def servo_finish(env, hist, max_steps=250):
    """Aproximacion final en lazo cerrado (graba demos)."""
    p = env.get_params()
    gx, gy, gth = goal_pose(p)
    dmax = p.vehicle.delta_max
    for _ in range(max_steps):
        x, y, th = env._state[:3]
        dg = math.hypot(gx - x, gy - y)
        tgt = gth if dg < 1.2 else math.atan2(gy - y, gx - x)
        err = wrap_angle(tgt - th)
        delta = float(np.clip(2.5 * err, -dmax, dmax))
        v = min(1.0, max(0.25, 0.3 + dg * 0.5))
        obs = env._get_obs().copy()
        _, _, term, trunc, info = env.step([v, delta])
        hist.append((obs, np.array([v, delta], dtype=np.float32)))
        if term or trunc:
            return info
    return info


def main():
    rng = np.random.default_rng(0)
    env = HeadlandTurnEnv(dr_mode="fixed", max_steps=600)
    for t in range(N_TRIALS):
        seq = [
            (VS[rng.integers(2)], DELTAS[rng.integers(len(DELTAS))], int(rng.integers(*DURS)))
            for _ in range(N_SEG)
        ]
        seq[0] = (0.8, seq[0][1], seq[0][2])  # primer segmento siempre avance
        hist = []
        info, early = rollout_open(env, seq, hist=hist)
        if early:
            continue  # colisiono o exito prematuro sin servo (raro): descartar
        info2 = servo_finish(env, hist)
        if t % 500 == 0:
            print(f"trial={t} dpos_open={info['dpos']:.2f} -> servo success={info2['success']} "
                  f"dpos={info2['dpos']:.2f} dth={info2['dtheta_deg']:.0f}")
        if info2["success"]:
            obs_arr = np.stack([h[0] for h in hist])
            act_arr = np.stack([h[1] for h in hist])
            np.save("data/demos/demo_A_obs.npy", obs_arr)
            np.save("data/demos/demo_A_act.npy", act_arr)
            th = np.arctan2(obs_arr[:, 3], obs_arr[:, 2])
            poses = np.column_stack([obs_arr[:, 0], obs_arr[:, 1], th])
            poses = np.vstack([poses, env._state[:3]])  # estado final
            np.save("data/references/reference_A.npy", poses)
            print(f"EXITO trial={t} steps={info2['steps']} rev={info2['rev_steps']}")
            print(f"demos: {len(hist)} pares (obs, act); referencia: {len(poses)} poses")
            return
    print("sin exito: ampliar N_TRIALS")


if __name__ == "__main__":
    main()
