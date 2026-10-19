"""Geometria de la maniobra: poses de inicio/goal, colisiones y factibilidad."""
from __future__ import annotations

import math
import numpy as np

from .params import EpisodeParams


def wrap_angle(a: float) -> float:
    """Envuelve un angulo a [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def start_pose(params: EpisodeParams) -> np.ndarray:
    """Pose inicial: salida de la calle origen, mirando a cabecera (+y).

    Calle origen en x=0, linea de cultivo en y=0.
    Retorna [x, y, theta] con theta=pi/2.
    """
    return np.array([0.0, 0.0, math.pi / 2.0], dtype=np.float64)


def goal_pose(params: EpisodeParams) -> np.ndarray:
    """Pose objetivo: entrada de la calle destino, mirando al cultivo (-y)."""
    x_goal = params.crop.n_lanes * params.crop.row_spacing
    return np.array([x_goal, 0.0, -math.pi / 2.0], dtype=np.float64)


def pose_errors(
    pose: np.ndarray, goal: np.ndarray
) -> tuple[float, float]:
    """Retorna (distancia euclidea [m], error angular absoluto [rad])."""
    dpos = float(np.hypot(pose[0] - goal[0], pose[1] - goal[1]))
    dtheta = abs(wrap_angle(float(pose[2]) - float(goal[2])))
    return dpos, dtheta


def nearest_alley_distance(x: float, row_spacing: float) -> float:
    """Distancia lateral al centro de la calle (alley) mas cercana."""
    k = round(x / row_spacing)
    return abs(x - k * row_spacing)


def in_crop_zone(y: float) -> bool:
    return y < 0.0


def check_collision(x: float, y: float, params: EpisodeParams) -> tuple[bool, str]:
    """Comprueba colision / salida de zona valida.

    Modelo v0 (simple y documentado):
    - Zona de cabecera 0 <= y <= H: libre, salvo limites exteriores
      x en [-x_margin, x_goal + x_margin], y <= H + y_margin_top.
    - Zona de cultivo y en [-crop_depth, 0): solo es libre el pasillo
      alrededor del centro de cada calle. Fuera del pasillo => colision
      con la fila de vinas.
    - Fuera de esos rangos => salida de zona valida ("out_of_bounds").

    Retorna (colision: bool, motivo: str).
    """
    crop = params.crop
    x_goal = crop.n_lanes * crop.row_spacing

    # Limite exterior general
    if y > crop.headland_width + crop.y_margin_top:
        return True, "out_of_bounds_top"
    if y < -crop.crop_depth:
        return True, "out_of_bounds_bottom"
    if x < -crop.x_margin or x > x_goal + crop.x_margin:
        return True, "out_of_bounds_side"

    # Cabecera: todo libre dentro de los limites
    if y >= 0.0:
        return False, ""

    # Cultivo: solo pasillos
    free_half = (crop.row_spacing - crop.row_width) / 2.0
    if nearest_alley_distance(x, crop.row_spacing) > free_half:
        return True, "vine_row"
    return False, ""


def min_distance_to_rows(x: float, params: EpisodeParams) -> float:
    """Distancia lateral al borde de fila de vinas mas cercano (>=0).

    Solo tiene sentido en zona de cultivo; en cabecera retorna inf.
    Util para la metrica 'distancia minima a filas'.
    """
    crop = params.crop
    free_half = (crop.row_spacing - crop.row_width) / 2.0
    d_center = nearest_alley_distance(x, crop.row_spacing)
    return max(0.0, free_half - d_center)


def feasibility_margin(params: EpisodeParams) -> float:
    """Margen de factibilidad geometrica (adimensional, v0).

    Compara la anchura de cabecera con el diametro minimo de giro
    necesario para un desplazamiento lateral de `n_lanes * spacing`.
    Regla simple: hacen falta ~2*R_min para un giro en U cerrada de
    una calle; para n calles, el requisito lateral es n*spacing.

    margen = H / (2 * R_min + n_lanes * spacing * 0.0) ... version simple:
    margen = H / (2*R_min). < 1 sugiere maniobra al limite/imposible
    en un solo arco (requeriria marcha atras o mas espacio).

    Es solo un indicador para el regimen 'frontera de factibilidad';
    la deteccion rigurosa de factibilidad (Dubins) llegara con el
    baseline clasico (fase 6).
    """
    r_min = params.vehicle.r_min
    if r_min <= 0:
        return 0.0
    return params.crop.headland_width / (2.0 * r_min)


def is_likely_infeasible(params: EpisodeParams) -> bool:
    """Heuristico v0: cabecera estrecha + radio grande + salto largo."""
    r_min = params.vehicle.r_min
    lateral = params.crop.n_lanes * params.crop.row_spacing
    # Si el diametro minimo supera claramente el espacio + desplazamiento
    return params.crop.headland_width < (r_min * 1.2) and lateral > r_min
