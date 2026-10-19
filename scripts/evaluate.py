"""Evaluacion de una politica entrenada (metricas principales del protocolo, §6).

Uso:
  uv run python scripts/evaluate.py --model runs/A/seed_0/ppo_final.zip --episodes 100
  uv run python scripts/evaluate.py --model runs/A/seed_0/best_model.zip --dr-mode fixed --stochastic
Guarda CSV por episodio + resumen JSON junto al modelo (o en --outdir).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import headland_rl  # noqa: F401
from headland_rl.envs.headland_env import HeadlandTurnEnv
from headland_rl.envs.params import CropParams, VehicleParams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--dr-mode", default="fixed")
    ap.add_argument("--stochastic", action="store_true",
                    help="acciones estocasticas (por defecto: deterministicas)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    from stable_baselines3 import PPO

    model = PPO.load(args.model)
    env = HeadlandTurnEnv(dr_mode=args.dr_mode)
    outdir = Path(args.outdir) if args.outdir else Path(args.model).parent
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done, path_len = False, 0.0
        prev_xy = obs[0:2].astype(float)
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _, term, trunc, info = env.step(action)
            path_len += float(np.hypot(obs[0] - prev_xy[0], obs[1] - prev_xy[1]))
            prev_xy = obs[0:2].astype(float)
            done = bool(term or trunc)
        p = env.get_params()
        rows.append({
            "ep": ep, "success": int(info["success"]),
            "collided": int(info["collided"]), "reason": info["collision_reason"],
            "dpos": info["dpos"], "dtheta_deg": info["dtheta_deg"],
            "steps": info["steps"], "path_len": path_len,
            "min_row_dist": info["min_row_dist"],
            "n_lanes": p.crop.n_lanes, "row_spacing": p.crop.row_spacing,
            "headland_width": p.crop.headland_width, "L": p.vehicle.L,
            "delta_max_deg": p.vehicle.delta_max_deg,
        })

    with open(outdir / "eval_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    a = np.array
    succ = [r for r in rows if r["success"]]
    summary = {
        "model": str(args.model), "dr_mode": args.dr_mode,
        "episodes": args.episodes, "deterministic": not args.stochastic,
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "collision_rate": float(np.mean([r["collided"] for r in rows])),
        "dpos_succ_mean": float(np.mean([r["dpos"] for r in succ])) if succ else None,
        "dpos_succ_std": float(np.std([r["dpos"] for r in succ])) if succ else None,
        "dtheta_succ_mean": float(np.mean([r["dtheta_deg"] for r in succ])) if succ else None,
        "steps_mean": float(np.mean([r["steps"] for r in rows])),
        "path_len_mean": float(np.mean([r["path_len"] for r in rows])),
        "min_row_dist_mean": float(np.mean([r["min_row_dist"] for r in rows if r["min_row_dist"] != float("inf")] or [float("nan")])),
    }
    with open(outdir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
