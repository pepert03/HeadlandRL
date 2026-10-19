"""Caja envolvente de los 6 tipos Dubins vs zona valida (factibilidad)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from headland_rl.envs import dubins as D
from headland_rl.envs.geometry import check_collision, goal_pose, start_pose
from headland_rl.envs.params import CropParams, EpisodeParams, VehicleParams

p = EpisodeParams(vehicle=VehicleParams(), crop=CropParams())
s0, g = start_pose(p), goal_pose(p)
alpha, beta, d = D._transform(*s0, *g, p.vehicle.r_min)
print(f"Rmin={p.vehicle.r_min:.2f} H={p.crop.headland_width} x_goal={g[0]:.2f}")
for name, fn in D._SOLVERS.items():
    sol = fn(alpha, beta, d)
    if sol is None:
        print(f"{name}: sin solucion")
        continue
    t, pp, q = sol
    L = (t + pp + q) * p.vehicle.r_min
    # reconstruye con el tipo forzado
    import headland_rl.envs.dubins as DD
    orig = DD._SOLVERS
    DD._SOLVERS = {name: fn}
    try:
        pts = np.array(DD.sample_dubins_path(*s0, *g, p.vehicle.r_min, ds=0.1))
    finally:
        DD._SOLVERS = orig
    bad = 0
    for x, y in pts[::5]:
        c, _ = check_collision(float(x), float(y), p)
        bad += int(c)
    print(f"{name}: L={L:.2f}m x=[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
          f"y=[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] pts_en_colision={bad}/{len(pts[::5])}")
