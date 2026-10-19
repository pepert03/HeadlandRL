import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from headland_rl.envs.dubins import dubins_path_length, sample_dubins_path


def _heading_along(pts, ds=0.1):
    dx = pts[-1][0] - pts[-5][0]
    dy = pts[-1][1] - pts[-5][1]
    return math.atan2(dy, dx)


def test_straight_line():
    L = dubins_path_length(0, 0, 0, 5, 0, 0, r_min=1.0)
    assert abs(L - 5.0) < 1e-9, L


def test_length_ge_euclidean_and_endpoint():
    rng = np.random.default_rng(0)
    for _ in range(60):
        x0, y0 = rng.uniform(-5, 5, 2)
        th0 = rng.uniform(-math.pi, math.pi)
        x1, y1 = rng.uniform(-5, 5, 2)
        th1 = rng.uniform(-math.pi, math.pi)
        r = rng.uniform(1.0, 4.0)
        L = dubins_path_length(x0, y0, th0, x1, y1, th1, r)
        eu = math.hypot(x1 - x0, y1 - y0)
        assert L + 1e-6 >= eu, (L, eu)
        pts = sample_dubins_path(x0, y0, th0, x1, y1, th1, r, ds=0.05)
        ex, ey = pts[-1]
        assert math.hypot(ex - x1, ey - y1) < 0.08, (ex, ey, x1, y1)
        th_end = _heading_along(pts)
        dth = abs((th_end - th1 + math.pi) % (2 * math.pi) - math.pi)
        # tolerancia Dominada por la discretizacion ds (cuerda vs arco)
        assert dth < 0.12, dth


def test_headland_case_uses_detour():
    # Caso del paso A: el Dubins debe ser mas largo que el euclideo (detour)
    L = dubins_path_length(0, 0, math.pi / 2, 2.5, 0, -math.pi / 2, r_min=3.47)
    eu = 2.5
    assert L > eu + 1.0, (L, eu)
    print(f"headland: euclideo={eu:.2f} dubins={L:.2f}")
