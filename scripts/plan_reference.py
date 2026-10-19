"""Referencia de backplay del caso A con Hybrid A* (huella completa).

Sustituye a `search_reference.py`: con la huella real del Twizy la busqueda
aleatoria open-loop ya no encuentra la maniobra. Busca sobre primitivas de
arco (adelante/atras, 5 angulos de direccion) comprobando colision con toda
la huella y penaliza marcha atras y cambios de sentido. Es cinematico: no
modela tau_v / tau_delta ni el tiempo de cambio de marcha.

Uso:
  uv run python scripts/plan_reference.py            # -> data/references/reference_A.npy
  uv run python scripts/plan_reference.py --H 4.5 --out /tmp/ref.npy
"""
from __future__ import annotations

import argparse
import heapq
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from headland_rl.envs.geometry import (
    check_footprint_collision,
    goal_pose,
    start_pose,
    wrap_angle,
)
from headland_rl.envs.params import CropParams, EpisodeParams, VehicleParams

STEP = 0.4  # longitud de cada primitiva [m]
SUB = 0.1  # paso de comprobacion de colision (y de la referencia) [m]
XY_RES = 0.15
TH_RES = math.radians(7.5)
REV_PEN = 1.5  # coste relativo de un metro marcha atras
SWITCH_PEN = 3.0  # coste de un cambio de sentido [m equivalentes]
GOAL_POS, GOAL_ANG = 0.2, math.radians(8.0)  # mas estricto que el exito del env


def plan(p: EpisodeParams, max_exp: int = 200_000):
    """Retorna (poses (N,3) cada SUB metros, n_expansiones) o (None, n)."""
    s0, g = start_pose(p), goal_pose(p)
    L, dmax = p.vehicle.L, p.vehicle.delta_max
    deltas = [-dmax, -dmax / 2, 0.0, dmax / 2, dmax]

    def key(x, y, th):
        return (round(x / XY_RES), round(y / XY_RES), round(wrap_angle(th) / TH_RES))

    def h(x, y, th):
        return math.hypot(g[0] - x, g[1] - y) + abs(wrap_angle(g[2] - th))

    start = (float(s0[0]), float(s0[1]), float(s0[2]))
    openq = [(h(*start), 0.0, start, 1, None, [start])]
    closed: dict = {}
    n = 0
    while openq and n < max_exp:
        _, cost, st, gear, par, trace = heapq.heappop(openq)
        k = key(*st)
        if k in closed:
            continue
        closed[k] = (par, trace)
        n += 1
        x, y, th = st
        if math.hypot(g[0] - x, g[1] - y) < GOAL_POS and abs(wrap_angle(g[2] - th)) < GOAL_ANG:
            segs = []
            while k is not None:
                par, trace = closed[k]
                segs.append(trace)
                k = par
            return np.array([q for seg in segs[::-1] for q in seg]), n
        for d in (1, -1):
            for de in deltas:
                xx, yy, tt = x, y, th
                trace = []
                for _ in range(int(round(STEP / SUB))):
                    xx += d * SUB * math.cos(tt)
                    yy += d * SUB * math.sin(tt)
                    tt = wrap_angle(tt + d * SUB / L * math.tan(de))
                    if check_footprint_collision(xx, yy, tt, p)[0]:
                        break
                    trace.append((xx, yy, tt))
                else:
                    if key(xx, yy, tt) in closed:
                        continue
                    c = cost + STEP * (REV_PEN if d < 0 else 1.0)
                    c += SWITCH_PEN if d != gear else 0.0
                    heapq.heappush(openq, (c + h(xx, yy, tt), c, (xx, yy, tt), d, k, trace))
    return None, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=float, default=None, help="anchura de cabecera (defecto: nominal)")
    ap.add_argument("--out", default="data/references/reference_A.npy")
    args = ap.parse_args()

    crop = CropParams() if args.H is None else CropParams(headland_width=args.H)
    p = EpisodeParams(vehicle=VehicleParams(), crop=crop)
    t0 = time.time()
    poses, n = plan(p)
    print(f"H={crop.headland_width} Rmin={p.vehicle.r_min:.2f} expansiones={n} "
          f"t={time.time() - t0:.0f}s")
    if poses is None:
        print("SIN SOLUCION")
        sys.exit(1)
    step = np.diff(poses[:, :2], axis=0)
    fwd = np.sign(step[:, 0] * np.cos(poses[:-1, 2]) + step[:, 1] * np.sin(poses[:-1, 2]))
    n_seg = 1 + int(np.sum(fwd[1:] != fwd[:-1]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, poses)
    print(f"{len(poses)} poses, {n_seg} segmentos, longitud {len(poses) * SUB:.1f} m -> {args.out}")


if __name__ == "__main__":
    main()
