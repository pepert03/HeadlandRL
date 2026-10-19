"""Curvas de Dubins: longitud minima y reconstruccion de la trayectoria.

Se usa para:
  - shaping de la recompensa RL: el progreso se mide sobre la longitud de
    Dubins al objetivo (que descuenta el detour necesario por la cabecera)
    en lugar de sobre la distancia euclidea (que lo castiga).
  - (fase 6) baseline clasico Dubins + Pure Pursuit: `dubins_solution`
    devuelve tambien los parametros (t, p, q) para reconstruir el camino.

Formulacion normalizada estandar (Shkel-Lumelsky): se traslada el problema
al marco del inicio y se escala por el radio minimo.
"""
from __future__ import annotations

import math

TWO_PI = 2.0 * math.pi


def _mod(a: float) -> float:
    return a % TWO_PI


def _transform(x0, y0, th0, x1, y1, th1, r_min):
    """Traslada/rota el objetivo al marco del inicio y escala por r_min."""
    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy) / r_min
    theta = math.atan2(dy, dx) % TWO_PI
    alpha = (th0 - theta) % TWO_PI
    beta = (th1 - theta) % TWO_PI
    return alpha, beta, d


def _lsl(alpha, beta, d):
    sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
    cab = math.cos(alpha - beta)
    tmp0 = d + sa - sb
    p_sq = 2.0 + d * d - 2.0 * cab + 2.0 * d * (sa - sb)
    if p_sq < 0:
        return None
    tmp1 = math.atan2(cb - ca, tmp0)
    t = _mod(-alpha + tmp1)
    p = math.sqrt(p_sq)
    q = _mod(beta - tmp1)
    return t, p, q


def _rsr(alpha, beta, d):
    sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
    cab = math.cos(alpha - beta)
    tmp0 = d - sa + sb
    p_sq = 2.0 + d * d - 2.0 * cab + 2.0 * d * (sb - sa)
    if p_sq < 0:
        return None
    tmp1 = math.atan2(ca - cb, tmp0)
    t = _mod(alpha - tmp1)
    p = math.sqrt(p_sq)
    q = _mod(-beta + tmp1)
    return t, p, q


def _lsr(alpha, beta, d):
    sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
    cab = math.cos(alpha - beta)
    p_sq = -2.0 + d * d + 2.0 * cab + 2.0 * d * (sa + sb)
    if p_sq < 0:
        return None
    p = math.sqrt(p_sq)
    tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
    t = _mod(-alpha + tmp)
    q = _mod(-beta + tmp)
    return t, p, q


def _rsl(alpha, beta, d):
    sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
    cab = math.cos(alpha - beta)
    p_sq = -2.0 + d * d + 2.0 * cab - 2.0 * d * (sa + sb)
    if p_sq < 0:
        return None
    p = math.sqrt(p_sq)
    tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
    t = _mod(alpha - tmp)
    q = _mod(beta - tmp)
    return t, p, q


def _rlr(alpha, beta, d):
    sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
    cab = math.cos(alpha - beta)
    tmp = (6.0 - d * d + 2.0 * cab + 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = _mod(TWO_PI - math.acos(tmp))
    t = _mod(alpha - math.atan2(ca - cb, d - sa + sb) + p / 2.0)
    q = _mod(alpha - beta - t + p)
    return t, p, q


def _lrl(alpha, beta, d):
    sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
    cab = math.cos(alpha - beta)
    tmp = (6.0 - d * d + 2.0 * cab + 2.0 * d * (sb - sa)) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = _mod(TWO_PI - math.acos(tmp))
    t = _mod(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
    q = _mod(beta - alpha - t + p)
    return t, p, q


_SOLVERS = {"LSL": _lsl, "RSR": _rsr, "LSR": _lsr, "RSL": _rsl, "RLR": _rlr, "LRL": _lrl}


def dubins_solution(x0, y0, th0, x1, y1, th1, r_min):
    """Mejor solucion Dubins: (tipo, t, p, q, longitud [m]).

    t, q: giros (rad, en unidades normalizadas por r_min); p: tramo central
    (recto en CSC, giro en CCC). Longitud = (t+p+q)*r_min.
    """
    alpha, beta, d = _transform(x0, y0, th0, x1, y1, th1, r_min)
    best = None
    for name, fn in _SOLVERS.items():
        sol = fn(alpha, beta, d)
        if sol is None:
            continue
        t, p, q = sol
        length = (t + p + q) * r_min
        if best is None or length < best[4]:
            best = (name, t, p, q, length)
    if best is None:
        raise RuntimeError("sin solucion Dubins (no deberia ocurrir en CSC)")
    return best


def dubins_path_length(x0, y0, th0, x1, y1, th1, r_min) -> float:
    """Longitud del camino Dubins mas corto [m]."""
    return dubins_solution(x0, y0, th0, x1, y1, th1, r_min)[4]


def dubins_reference_poses(x0, y0, th0, x1, y1, th1, r_min, ds=0.1):
    """Poses (x, y, th) equiespaciadas ~ds a lo largo del mejor camino Dubins.

    La ultima pose es exactamente el objetivo con su rumbo. Se usa para el
    backplay (spawn a lo largo de la referencia) y para la fase 6.
    """
    pts = sample_dubins_path(x0, y0, th0, x1, y1, th1, r_min, ds=ds)
    poses = []
    for i in range(len(pts)):
        j = min(i + 1, len(pts) - 1)
        k = max(i - 1, 0)
        th = math.atan2(pts[j][1] - pts[k][1], pts[j][0] - pts[k][0])
        poses.append((pts[i][0], pts[i][1], th))
    poses[-1] = (x1, y1, th1)
    return poses


def sample_dubins_path(x0, y0, th0, x1, y1, th1, r_min, ds=0.1):
    """Reconstruye puntos (x, y) del mejor camino Dubins cada `ds` metros."""
    name, t, p, q, _ = dubins_solution(x0, y0, th0, x1, y1, th1, r_min)
    if name in ("LSL", "RSR", "LSR", "RSL"):
        segs = [(name[0], t), ("S", p), (name[2], q)]
    else:  # RLR / LRL: tramo central curvo de sentido opuesto
        mid = "L" if name == "RLR" else "R"
        segs = [(name[0], t), (mid, p), (name[2], q)]
    pts = [(x0, y0)]
    x, y, th = x0, y0, th0
    for kind, plen in segs:
        s = 0.0
        while s < plen * r_min - 1e-9:
            step = min(ds, plen * r_min - s)
            if kind == "S":
                x += step * math.cos(th)
                y += step * math.sin(th)
            else:
                turn = 1.0 if kind == "L" else -1.0
                dth = turn * step / r_min
                # integracion exacta del arco
                x += r_min * (math.sin(th + dth) - math.sin(th)) * turn
                y += r_min * (-math.cos(th + dth) + math.cos(th)) * turn
                th += dth
            s += step
            pts.append((x, y))
    return pts
