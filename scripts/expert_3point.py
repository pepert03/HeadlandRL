"""Experto guionado de 3 puntos para el caso A (scaffolding de demos/backplay).

Fases de servo de rumbo + aproximacion final al objetivo. Determinista:
sirve para generar la referencia de backplay y (si hace falta) demos de BC.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math
import numpy as np

from headland_rl.envs.geometry import goal_pose, start_pose, wrap_angle
from headland_rl.envs.headland_env import HeadlandTurnEnv

# (v, rumbo_objetivo_deg, tol_deg, max_steps) ; rumbo None = fase final al goal
PHASES = [
    (1.0, 140.0, 6.0, 150),    # P1: avance girando a izquierdas
    (-0.8, 235.0, 8.0, 200),   # P2: retroceso barriendo el morro a 235
    (1.0, 270.0, 6.0, 150),    # P3: avance alineando a -90 (270)
    ("goal", None, None, 200),  # P4: pursuit final al objetivo
]


def servo(env, v, target_deg, tol_deg, max_steps, trace):
    """Servo de rumbo. Retorna True/False si el episodio termina (exito o no),
    None si la fase completa sin terminar el episodio."""
    dmax = env.get_params().vehicle.delta_max
    for _ in range(max_steps):
        x, y, th = env._state[:3]
        err = wrap_angle(math.radians(target_deg) - th)
        if abs(math.degrees(err)) < tol_deg:
            return None
        sgn = 1.0 if v >= 0 else -1.0  # en reversa el servo invierte el signo
        delta = float(np.clip(2.0 * sgn * err, -dmax, dmax))
        _, _, term, trunc, info = env.step([v, delta])
        trace.append(tuple(env._state[:3]))
        if term or trunc:
            return bool(info["success"])
    return None


def final_approach(env, max_steps, trace):
    p = env.get_params()
    gx, gy, gth = goal_pose(p)
    dmax = p.vehicle.delta_max
    for _ in range(max_steps):
        x, y, th = env._state[:3]
        dg = math.hypot(gx - x, gy - y)
        if dg < 1.2:
            tgt = gth
        else:
            tgt = math.atan2(gy - y, gx - x)
        err = wrap_angle(tgt - th)
        delta = float(np.clip(2.5 * err, -dmax, dmax))
        v = min(1.0, max(0.25, 0.3 + dg * 0.5))
        _, _, term, trunc, info = env.step([v, delta])
        trace.append(tuple(env._state[:3]))
        if term or trunc:
            return bool(info["success"])
    return None


def run_full(seed=0, max_steps=600):
    """Ejecuta las 4 fases. Retorna (exito, env, traza)."""
    env = HeadlandTurnEnv(dr_mode="fixed", max_steps=max_steps)
    env.reset(seed=seed)
    trace = [tuple(env._state[:3])]
    for ph in PHASES:
        if env._steps >= max_steps:
            return False, env, trace
        if ph[0] == "goal":
            r = final_approach(env, ph[3], trace)
        else:
            r = servo(env, ph[0], ph[1], ph[2], ph[3], trace)
        if r is True:
            return True, env, trace
        if r is False:
            return False, env, trace
    return False, env, trace


def main():
    for seed in range(5):
        ok, env, _ = run_full(seed)
        x, y, th = env._state[:3]
        print(f"seed={seed} exito={ok} x={x:.2f} y={y:.2f} th={math.degrees(th):.0f} "
              f"steps={env._steps} rev={env._rev_steps}")
    ok, _, ref = run_full(0)
    print("OK" if ok else "FALLO")
    if ok:
        np.save("data/references/reference_A_expert3p.npy", np.array(ref))
        print(f"referencia guardada: data/references/reference_A_expert3p.npy ({len(ref)} poses)")


if __name__ == "__main__":
    main()
