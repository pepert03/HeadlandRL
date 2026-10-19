"""Dibuja la referencia Dubins del caso A (diagnostico / fase 6)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from headland_rl.envs.dubins import dubins_solution, sample_dubins_path
from headland_rl.envs.geometry import goal_pose, start_pose
from headland_rl.envs.params import CropParams, EpisodeParams, VehicleParams

p = EpisodeParams(vehicle=VehicleParams(), crop=CropParams())
s0, g = start_pose(p), goal_pose(p)
name, t, pp, q, L = dubins_solution(*s0, *g, p.vehicle.r_min)
print(f"tipo={name} t={t:.2f} p={pp:.2f} q={q:.2f} L={L:.2f}m Rmin={p.vehicle.r_min:.2f}")
pts = np.array(sample_dubins_path(*s0, *g, p.vehicle.r_min, ds=0.1))
print(f"y_min={pts[:,1].min():.2f} (cabecera: y>=0), n_pts={len(pts)}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.add_patch(plt.Rectangle((-2, 0), 6.5, 6.0, color="#f5f0dd", alpha=0.5))
ax.plot(pts[:, 0], pts[:, 1], "r-", lw=1.5, label=f"Dubins {name} L={L:.1f}m")
ax.plot([s0[0]], [s0[1]], "go", label="inicio")
ax.plot([g[0]], [g[1]], "r*", ms=10, label="objetivo")
ax.set_aspect("equal")
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig("runs/dubins_ref.png")
print("saved runs/dubins_ref.png")
