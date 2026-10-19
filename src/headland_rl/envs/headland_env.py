"""Entorno Gymnasium HeadlandTurn-v0 (version base, sin modelos entrenados).

Modelo (del report):
  estado: pose (x, y, theta) + actuadores (v, delta)
  entradas: v_cmd, delta_cmd
  x_dot = v cos(theta), y_dot = v sin(theta), theta_dot = (v/L) tan(delta)
  v_dot = (v_cmd - v)/tau_v, delta_dot = (delta_cmd - delta)/tau_delta
  |delta| <= delta_max, -v_rev_max <= v <= v_max, R_min = L/tan(delta_max)
  (el report dice 0 <= v; se admite marcha atras porque el caso nominal
  no tiene maniobra solo-avance, ver VehicleParams.v_rev_max)

Geometria:
  - calle origen en x=0, destino en x=n_lanes*row_spacing
  - linea de cultivo en y=0; cabecera en 0<=y<=H; cultivo en y<0
  - inicio (0,0,pi/2), objetivo (x_goal,0,-pi/2)

La recompensa es v3 (progreso euclideo + terminales dominantes).
Se documenta en la fase 4 (definicion de recompensa) del plan.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .geometry import (
    check_footprint_collision,
    feasibility_margin,
    footprint_min_row_distance,
    footprint_points,
    goal_pose,
    is_likely_infeasible,
    pose_errors,
    start_pose,
    wrap_angle,
)
from .dubins import dubins_path_length, dubins_reference_poses
from .params import CropParams, DomainRanges, EpisodeParams, InitNoise, VehicleParams
from .randomization import DRMode, sample_episode_params


DEFAULT_DELTA_ABS_DEG = 35.0  # techo del espacio de accion (fijo p/ DR)
V_STOP = 0.02  # [m/s] por debajo se considera parado (cambio de marcha)


class HeadlandTurnEnv(gym.Env):
    """Entorno parametrizable de cambio de calle (headland turn)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        dr_mode: DRMode | str = DRMode.FIXED,
        privileged: bool = False,
        dt: float = 0.05,
        max_steps: int = 1000,
        fixed_vehicle: VehicleParams | None = None,
        fixed_crop: CropParams | None = None,
        ranges: DomainRanges | None = None,
        init_noise: InitNoise | None = None,
        pos_tol: float = 0.30,
        ang_tol_deg: float = 10.0,
        curr_pos_0: float = 2.0,
        curr_ang_0_deg: float = 90.0,
        curr_speed_0: float = 0.4,
        curr_speed_1: float = 0.83,
        # pesos de recompensa v3 (fase 4): progreso euclideo + terminales.
        # (v2 usaba el progreso Dubins, que castiga la marcha atras; ver step)
        w_progress: float = 3.0,
        w_angle: float = 0.0,
        w_time: float = 0.01,
        w_smooth: float = 0.005,
        success_bonus: float = 100.0,
        collision_penalty: float = -25.0,
        # bonus intermedio (una vez por episodio) por entrar en una zona
        # amplia alrededor del objetivo: guia el descubrimiento sin cambiar
        # la definicion estricta de exito (pos_tol / ang_tol).
        zone_bonus: float = 10.0,
        zone_pos: float = 1.0,
        zone_ang_deg: float = 45.0,
        v_max_cap: float | None = None,
        reference_poses: np.ndarray | list | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.dr_mode = DRMode(dr_mode)
        self.privileged = privileged
        self.dt = dt
        self.max_steps = max_steps
        self.fixed_vehicle = fixed_vehicle or VehicleParams()
        self.fixed_crop = fixed_crop or CropParams()
        self.ranges = ranges or DomainRanges()
        self.init_noise = init_noise or InitNoise()
        self.pos_tol = pos_tol
        self.ang_tol = math.radians(ang_tol_deg)
        # Tolerancias estrictas (metrica del protocolo) + punto de partida
        # del curriculum (solo entrenamiento; la evaluacion usa las estrictas).
        self._pos_tol_strict = pos_tol
        self._ang_tol_strict = math.radians(ang_tol_deg)
        self._curr_pos_0 = curr_pos_0
        self._curr_ang_0 = math.radians(curr_ang_0_deg)
        self._cur_pos = pos_tol
        self._cur_ang = math.radians(ang_tol_deg)
        # Backplay (fase 4): con curriculum activo, el spawn se muestrea
        # U(u_min, u_max) a lo largo de la referencia (mezcla con ensayo:
        # el suelo u_min recede pero siempre se repasan tramos faciles).
        # Sin curriculum (evaluacion): siempre inicio real.
        self._curr_on = False
        self._u_min = 0.0
        self._u_max = 1.0
        self._frontier_p = 0.8  # fraccion de episodios en la frontera (resto: ensayo)
        self.w_progress = w_progress
        self.w_angle = w_angle
        self.w_time = w_time
        self.w_smooth = w_smooth
        self.success_bonus = success_bonus
        self.collision_penalty = collision_penalty
        self.zone_bonus = zone_bonus
        self.zone_pos = zone_pos
        self.zone_ang = math.radians(zone_ang_deg)
        self._zone_given = False
        # Tope de velocidad de entrenamiento (curriculum): limita v_cmd por
        # debajo de vehicle.v_max al inicio para que la exploracion aleatoria
        # sobreviva lo suficiente como para descubrir la maniobra.
        self._v_cap = v_max_cap
        self._speed_0 = curr_speed_0
        self._speed_1 = curr_speed_1
        # Referencia experta opcional para backplay (p.ej. 3 puntos): si se da,
        # el spawn usa estas poses en lugar de la referencia Dubins.
        self._ref_override = None if reference_poses is None else np.asarray(reference_poses, dtype=np.float64)
        self.render_mode = render_mode

        # Espacio de accion FIJO (para que SB3 no cambie entre episodios DR):
        # [v_cmd en [-v_rev_max, v_max], delta_cmd en [-35, +35] deg].
        # v<0 = marcha atras (imprescindible: el caso nominal no admite
        # maniobra solo-avance). Se recorta a los limites del episodio.
        # v_max y v_rev_max no se aleatorizan: salen del vehiculo fijo.
        d_abs = math.radians(DEFAULT_DELTA_ABS_DEG)
        v_hi = self.fixed_vehicle.v_max
        v_lo = -self.fixed_vehicle.v_rev_max
        self.action_space = spaces.Box(
            low=np.array([v_lo, -d_abs], dtype=np.float32),
            high=np.array([v_hi, d_abs], dtype=np.float32),
            dtype=np.float32,
        )

        # Observacion base (12 dims):
        # [x, y, cos(th), sin(th), v, delta, dx_goal, dy_goal, cos(dth), sin(dth),
        #  marcha (+1/-1), progreso del cambio de marcha en curso [0,1]]
        # Si privileged=True se concatenan 7 params reales normalizados [0,1].
        base_low = np.array(
            [-10, -10, -1, -1, v_lo, -d_abs, -20, -20, -1, -1, -1, 0], dtype=np.float32
        )
        base_high = np.array(
            [20, 20, 1, 1, v_hi, d_abs, 20, 20, 1, 1, 1, 1], dtype=np.float32
        )
        if privileged:
            low = np.concatenate([base_low, np.zeros(7, dtype=np.float32)])
            high = np.concatenate([base_high, np.ones(7, dtype=np.float32)])
        else:
            low, high = base_low, base_high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._rng: np.random.Generator = np.random.default_rng()
        self.params: EpisodeParams = EpisodeParams(
            vehicle=self.fixed_vehicle, crop=self.fixed_crop
        )
        self._state = np.zeros(5, dtype=np.float64)  # x, y, th, v, delta
        self._goal = goal_pose(self.params)
        self._steps = 0
        self._prev_action = np.zeros(2, dtype=np.float64)
        self._trail: list[tuple[float, float]] = []
        self._min_row_dist = math.inf
        self._gear = 1  # +1 adelante, -1 atras
        self._gear_timer = 0.0
        self._n_switches = 0

    # ------------------------------------------------------------------ API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        opts = options or {}

        # Permite forzar params concretos (util en evaluacion)
        if "params" in opts and isinstance(opts["params"], EpisodeParams):
            self.params = opts["params"]
        else:
            self.params = sample_episode_params(
                self.dr_mode, self._rng,
                self.fixed_vehicle, self.fixed_crop, self.ranges,
            )

        s0 = start_pose(self.params)
        rand_geom = self.dr_mode in (DRMode.GEOMETRY, DRMode.JOINT)
        if rand_geom and not self._curr_on:
            s0 = s0 + np.array(
                [
                    self._rng.normal(0, self.init_noise.x_std),
                    self._rng.normal(0, self.init_noise.y_std),
                    math.radians(
                        self._rng.normal(0, self.init_noise.theta_std_deg)
                    ),
                ]
            )
        self._goal = goal_pose(self.params)
        if self._curr_on:
            if self._rng.random() < self._frontier_p:
                u = self._u_min
            else:
                u = float(self._rng.uniform(self._u_min, self._u_max))
            if u > 1e-6:
                s0 = self._backplay_pose(s0, u)
        self._state = np.array([s0[0], s0[1], s0[2], 0.0, 0.0], dtype=np.float64)
        self._prev_dpos, self._prev_dtheta = pose_errors(self._state[:3], self._goal)
        self._prev_dub = self._dubins_to_goal()
        self._zone_given = False
        self._rev_steps = 0  # pasos en marcha atras (metrica 11: segmentos)
        self._gear = 1
        self._gear_timer = 0.0
        self._n_switches = 0  # cambios de marcha (metrica 11: segmentos)
        self._steps = 0
        self._prev_action = np.zeros(2)
        self._trail = [(float(s0[0]), float(s0[1]))]
        self._min_row_dist = math.inf

        obs = self._get_obs()
        info = {"params": asdict(self.params.vehicle) | asdict(self.params.crop)}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        assert action.shape == (2,), f"accion esperada (2,), recibida {action.shape}"

        v_lim = self.params.vehicle.v_max
        if self._v_cap is not None:
            v_lim = min(v_lim, self._v_cap)
        v_rev = self.params.vehicle.v_rev_max
        v_cmd = float(np.clip(action[0], -v_rev, v_lim))
        d_lim = self.params.vehicle.delta_max
        delta_cmd = float(np.clip(action[1], -d_lim, d_lim))

        # Cambio de marcha: si la orden pide el sentido contrario a la marcha
        # engranada, el vehiculo frena hasta parar y espera gear_switch_time
        # antes de engranar la nueva (en el Twizy la marcha se cambia por CAN
        # emulando los botones). Mientras tanto la consigna de velocidad es 0.
        x, y, th, v, delta = self._state
        want = 1 if v_cmd > 1e-3 else (-1 if v_cmd < -1e-3 else 0)
        v_target = v_cmd
        if want != 0 and want != self._gear:
            v_target = 0.0
            if abs(v) < V_STOP:
                self._gear_timer += self.dt
                if self._gear_timer >= self.params.vehicle.gear_switch_time - 1e-9:
                    self._gear = want
                    self._gear_timer = 0.0
                    self._n_switches += 1
        else:
            self._gear_timer = 0.0

        # Dinamica de primer orden (Euler explicito)
        tau_v = self.params.vehicle.tau_v
        tau_d = self.params.vehicle.tau_delta
        v = v + (v_target - v) * (self.dt / max(tau_v, 1e-6))
        delta = delta + (delta_cmd - delta) * (self.dt / max(tau_d, 1e-6))
        delta = float(np.clip(delta, -d_lim, d_lim))
        v = float(np.clip(v, -v_rev, self.params.vehicle.v_max))

        L = self.params.vehicle.L
        x = x + v * math.cos(th) * self.dt
        y = y + v * math.sin(th) * self.dt
        th = th + (v / L) * math.tan(delta) * self.dt
        th = wrap_angle(th)
        self._state = np.array([x, y, th, v, delta])
        self._steps += 1
        if v < -0.05:
            self._rev_steps += 1
        self._trail.append((x, y))

        dpos, dtheta = pose_errors(self._state[:3], self._goal)
        collided, reason = check_footprint_collision(x, y, th, self.params)
        self._min_row_dist = min(
            self._min_row_dist, footprint_min_row_distance(x, y, th, self.params)
        )

        success = (dpos < self._cur_pos) and (dtheta < self._cur_ang)
        timeout = self._steps >= self.max_steps
        terminated = bool(success or collided)
        truncated = bool((not terminated) and timeout)

        # Recompensa v3 (fase 4): progreso EUCLIDEO + terminales.
        # Con marcha atras la maniobra es en 3 puntos sin detour: el euclideo
        # es monotono y el potencial Dubins (solo-avance) castigaria la
        # reversa. El Dubins se conserva solo como metrica en info.
        d_smooth = float(np.sum((np.array([v_cmd, delta_cmd]) - self._prev_action) ** 2))
        self._prev_action = np.array([v_cmd, delta_cmd])
        dub = self._dubins_to_goal()
        reward = (
            self.w_progress * (self._prev_dpos - dpos)
            + self.w_angle * (self._prev_dtheta - dtheta) / math.pi
            - self.w_time
            - self.w_smooth * d_smooth
        )
        self._prev_dpos, self._prev_dtheta, self._prev_dub = dpos, dtheta, dub
        if (not self._zone_given) and dpos < self.zone_pos and dtheta < self.zone_ang:
            reward += self.zone_bonus
            self._zone_given = True
        if success:
            reward += self.success_bonus
        if collided:
            # Coste de oportunidad: se imputa el tiempo no vivido para que
            # chocar pronto no ahorre penalizacion temporal (fase 4). El exito
            # rapido si conserva su ventaja (solo paga el tiempo vivido).
            reward += self.collision_penalty - self.w_time * (self.max_steps - self._steps)

        info = {
            "dpos": dpos,
            "dtheta_deg": math.degrees(dtheta),
            "dubins_len": dub,
            "success": success,
            "collided": collided,
            "collision_reason": reason,
            "timeout": timeout,
            "steps": self._steps,
            "min_row_dist": self._min_row_dist,
            "feasibility_margin": feasibility_margin(self.params),
            "likely_infeasible": is_likely_infeasible(self.params),
            "R_min": self.params.vehicle.r_min,
            "x_goal": float(self._goal[0]),
            "rev_steps": self._rev_steps,
            "gear_switches": self._n_switches,
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------- helpers
    def _get_obs(self) -> np.ndarray:
        x, y, th, v, delta = self._state
        dx, dy = float(self._goal[0] - x), float(self._goal[1] - y)
        dth = wrap_angle(float(self._goal[2]) - th)
        switch = self._gear_timer / max(self.params.vehicle.gear_switch_time, 1e-6)
        base = np.array(
            [x, y, math.cos(th), math.sin(th), v, delta,
             dx, dy, math.cos(dth), math.sin(dth), float(self._gear), switch],
            dtype=np.float32,
        )
        if not self.privileged:
            return base
        mu = np.array(self.params.as_privileged_vector(), dtype=np.float32)
        mu_norm = self._normalize_privileged(mu)
        return np.concatenate([base, mu_norm]).astype(np.float32)

    def _normalize_privileged(self, mu: np.ndarray) -> np.ndarray:
        """Normaliza mu a [0,1] usando los rangos de entrenamiento."""
        r = self.ranges
        bounds = [
            r.row_spacing, r.headland_width,
            (float(r.n_lanes[0]), float(r.n_lanes[1])),
            r.L, r.delta_max_deg, r.tau_v, r.tau_delta,
        ]
        out = np.zeros_like(mu, dtype=np.float32)
        for i, ((lo, hi), val) in enumerate(zip(bounds, mu)):
            out[i] = float(np.clip((val - lo) / (hi - lo + 1e-9), 0.0, 1.0))
        return out

    def _dubins_to_goal(self) -> float:
        """Longitud de Dubins del estado actual al objetivo (potencial v2)."""
        x, y, th = self._state[:3]
        gx, gy, gth = self._goal
        return dubins_path_length(x, y, th, gx, gy, float(gth),
                                  self.params.vehicle.r_min)

    def get_params(self) -> EpisodeParams:
        return self.params

    def set_curriculum(self, frac: float) -> None:
        """Endurece tolerancias y tope de velocidad linealmente (fase 4).

        frac=0 -> (curr_pos_0, curr_ang_0, speed_0); frac>=1 -> valores
        estrictas. Se llama desde el callback de entrenamiento sobre el
        entorno de rollout; la evaluacion siempre usa los valores estrictos.
        """
        f = float(min(max(frac, 0.0), 1.0))
        self._cur_pos = self._pos_tol_strict + (self._curr_pos_0 - self._pos_tol_strict) * (1.0 - f)
        self._cur_ang = self._ang_tol_strict + (self._curr_ang_0 - self._ang_tol_strict) * (1.0 - f)
        self._v_cap = self._speed_0 + (self._speed_1 - self._speed_0) * f
        self._curr_on = True
        self._u_min = 1.0 - f
        self._u_max = 1.0

    def _backplay_pose(self, s0_nominal: np.ndarray, u: float) -> np.ndarray:
        """Pose de spawn a fraccion `u` del arco de la referencia.

        Usa la referencia experta si se dio (`reference_poses`) o la Dubins
        en su defecto. Si el punto cae en colision, se avanza al objetivo
        hasta una pose libre (huella completa).
        """
        if self._ref_override is not None:
            poses = [tuple(p) for p in self._ref_override]
        else:
            gx, gy, gth = self._goal
            poses = dubins_reference_poses(
                float(s0_nominal[0]), float(s0_nominal[1]), float(s0_nominal[2]),
                float(gx), float(gy), float(gth),
                self.params.vehicle.r_min, ds=0.1,
            )
        idx = min(int(u * (len(poses) - 1)), len(poses) - 1)
        while idx < len(poses) - 1:
            x, y, th = poses[idx]
            collided, _ = check_footprint_collision(x, y, th, self.params)
            if not collided:
                break
            idx += 1
        x, y, th = poses[idx]
        return np.array([x, y, th], dtype=np.float64)

    def set_speed_cap(self, v: float) -> None:
        self._v_cap = float(v)

    def get_goal(self) -> np.ndarray:
        return self._goal.copy()

    # ------------------------------------------------------------- render
    def render(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        crop = self.params.crop
        x_goal = float(self._goal[0])
        H = crop.headland_width
        fig, ax = plt.subplots(figsize=(7, 5))
        # Cabecera
        ax.add_patch(
            plt.Rectangle((-crop.x_margin, 0), x_goal + 2 * crop.x_margin, H,
                          color="#f5f0dd", alpha=0.6, label="cabecera")
        )
        # Filas de vinas (zona y<0): lineas verdes en bordes de pasillos
        xs = np.arange(-1, crop.n_lanes + 2) * crop.row_spacing
        for xc in xs:
            for edge in (-crop.row_width / 2, crop.row_width / 2):
                ax.plot([xc + edge - crop.row_spacing / 2] * 2,
                        [-crop.crop_depth, 0], color="green", lw=2, alpha=0.7)
        # Inicio / objetivo
        s0 = start_pose(self.params)
        ax.plot(s0[0], s0[1], "bo", label="inicio")
        ax.plot(self._goal[0], self._goal[1], "r*", ms=12, label="objetivo")
        # Trayectoria
        if len(self._trail) > 1:
            tr = np.array(self._trail)
            ax.plot(tr[:, 0], tr[:, 1], "b-", lw=1.5)
        x, y, th = self._state[:3]
        fp = footprint_points(x, y, th, self.params, ds=0.5)
        ax.fill(fp[:, 0], fp[:, 1], color="blue", alpha=0.25)
        ax.arrow(x, y, 0.4 * math.cos(th), 0.4 * math.sin(th),
                 head_width=0.15, color="blue")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(
            f"HeadlandTurn ({self.dr_mode.value}"
            f"{', privilegiado' if self.privileged else ''}) "
            f"H={H:.1f}m d={crop.n_lanes}x{crop.row_spacing:.2f}m"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return buf
