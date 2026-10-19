"""Baseline clasico (fase 6): Dubins + Pure Pursuit.

Tambien sirve como experto para el warm-start BC del paso A (idea de
Pindl et al. 2024: Dubins como conocimiento experto).
"""
from __future__ import annotations

import math

import numpy as np

from ..envs.dubins import dubins_reference_poses
from ..envs.geometry import wrap_angle


class DubinsPurePursuit:
    """Sigue la referencia Dubins (recalculada por episodio) con Pure Pursuit.

    Entrada: pose (x, y, th). Salida: (v_cmd, delta_cmd).
    """

    def __init__(
        self,
        lookahead: float = 0.8,
        v_max: float = 2.0,
        v_min: float = 0.4,
        slow_radius: float = 3.0,
    ):
        self.lookahead = lookahead
        self.v_max = v_max
        self.v_min = v_min
        self.slow_radius = slow_radius
        self._ref: list[tuple[float, float, float]] = []
        self._goal = np.zeros(3)
        self._L = 1.69
        self._dmax = math.radians(26.0)

    def reset_episode(self, params, start, goal) -> None:
        self._L = params.vehicle.L
        self._dmax = params.vehicle.delta_max
        self._v_max = min(self.v_max, params.vehicle.v_max)
        self._goal = np.array(goal, dtype=float)
        self._ref = dubins_reference_poses(
            *start, *goal, params.vehicle.r_min, ds=0.1
        )

    def act(self, x: float, y: float, th: float) -> np.ndarray:
        pts = np.array([(p[0], p[1]) for p in self._ref])
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        i0 = int(np.argmin(d))
        # punto lookahead: primer punto a >= Ld del vehiculo, por delante
        Ld = self.lookahead
        i = i0
        while i < len(pts) - 1 and math.hypot(pts[i][0] - x, pts[i][1] - y) < Ld:
            i += 1
        lx, ly = pts[i]
        alpha = wrap_angle(math.atan2(ly - y, lx - x) - th)
        delta = math.atan2(2.0 * self._L * math.sin(alpha), Ld)
        delta = float(np.clip(delta, -self._dmax, self._dmax))
        dg = math.hypot(self._goal[0] - x, self._goal[1] - y)
        v = self._v_max * min(1.0, max(self.v_min / self._v_max, dg / self.slow_radius))
        v *= 1.0 / (1.0 + 2.0 * abs(delta) / self._dmax)
        return np.array([max(v, self.v_min * 0.5), delta])
