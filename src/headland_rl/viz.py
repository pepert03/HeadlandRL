"""Dibujo vectorial de la escena (estilo ilustracion plana, vista cenital).

Paleta y formas inspiradas en los esquemas de headland turn de tractor:
cabecera gris, calles color tierra, bordes marron oscuro, trayectoria roja.
El vehiculo se dibuja a la escala de su huella (eje trasero = origen local,
x hacia delante) y las ruedas delanteras giran con el angulo de direccion.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.transforms import Affine2D

from .envs.params import EpisodeParams, VehicleParams

PALETTE = {
    "headland": "#bdbdbd",
    "field": "#c9afa1",
    "border": "#5a1e0a",
    "row": "#7a4a32",
    "vine": "#4f7a3a",
    "vine_dark": "#3c5f2c",
    "path": "#c0392b",
    "ref": "#e8a39b",
    "tyre": "#2b2216",
    "body": "#c0393b",
    "body_dark": "#8e2a2c",
    "cabin": "#2e3a78",
    "seat": "#1a1a1a",
    "metal": "#7d7d7d",
}


def draw_scene(ax, params: EpisodeParams, x_goal: float) -> None:
    """Cabecera, calles, filas de vina y bordes (estaticos)."""
    crop = params.crop
    s, H = crop.row_spacing, crop.headland_width
    x0, x1 = -crop.x_margin, x_goal + crop.x_margin
    y0, y1 = -crop.crop_depth, H + crop.y_margin_top
    ax.set_facecolor("white")
    ax.add_patch(Rectangle((x0, 0), x1 - x0, y1, color=PALETTE["headland"], zorder=0, lw=0))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, -y0, color=PALETTE["field"], zorder=0, lw=0))

    # filas de vina entre calles: banda oscura + cepas
    k0, k1 = math.floor(x0 / s), math.ceil(x1 / s)
    for k in range(k0, k1):
        xc = (k + 0.5) * s
        lo, hi = max(xc - crop.row_width / 2, x0), min(xc + crop.row_width / 2, x1)
        if hi <= lo:
            continue
        ax.add_patch(Rectangle((lo, y0), hi - lo, -y0, color=PALETTE["row"], zorder=1, lw=0))
        if lo > x0 and hi < x1:
            for yv in np.arange(-0.5, y0, -1.0):
                r = crop.row_width * 0.45
                ax.add_patch(Circle((xc, yv), r, color=PALETTE["vine"], zorder=2, lw=0))
                ax.add_patch(Circle((xc + r * 0.25, yv - r * 0.2), r * 0.45,
                                    color=PALETTE["vine_dark"], zorder=2, lw=0))

    lw = 3.0
    ax.plot([x0, x1], [0, 0], color=PALETTE["border"], lw=lw, zorder=3, solid_capstyle="butt")
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           ec=PALETTE["border"], lw=lw, zorder=3))
    ax.set_xlim(x0 - 0.15, x1 + 0.15)
    ax.set_ylim(y0 - 0.15, y1 + 0.15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


class VehicleArtist:
    """Vehiculo vectorial desde arriba; `update(x, y, th, delta)` lo mueve."""

    def __init__(self, ax, veh: VehicleParams, zorder: int = 6, alpha: float = 1.0,
                 ghost: bool = False):
        self.ax = ax
        L, W = veh.L, veh.width
        hw, xr, xf = W / 2, -veh.rear_overhang, veh.L + veh.front_overhang
        c = PALETTE
        self._parts: list[tuple[object, tuple[float, float] | None]] = []

        def box(x, y, w, h, color, z, r=0.04, steer=None):
            if ghost:
                p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                   fc="none", ec=PALETTE["path"], lw=1.2, ls=(0, (3, 2)),
                                   alpha=alpha, zorder=z)
            else:
                p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                   fc=color, ec="none", alpha=alpha, zorder=z)
            ax.add_patch(p)
            self._parts.append((p, steer))

        if ghost:  # solo el contorno de la huella
            box(xr, -hw, xf - xr, W, None, zorder, r=0.12)
            return

        rw_l, rw_w = min(0.66, 0.45 * L), 0.28  # rueda trasera
        fw_l, fw_w = min(0.46, 0.30 * L), 0.19  # rueda delantera
        ry, fy = hw - rw_w / 2, hw - fw_w / 2 - 0.06
        # enganche trasero
        box(xr, -0.12, 0.16, 0.24, c["metal"], zorder, r=0.03)
        # eje delantero
        box(L - 0.03, -fy, 0.06, 2 * fy, c["tyre"], zorder, r=0.02)
        # ruedas
        for sgn in (-1, 1):
            box(-rw_l / 2, sgn * ry - rw_w / 2, rw_l, rw_w, c["tyre"], zorder + 1, r=0.07)
            box(L - fw_l / 2, sgn * fy - fw_w / 2, fw_l, fw_w, c["tyre"], zorder + 1, r=0.06,
                steer=(L, sgn * fy))
        # capo (de la cabina al frontal)
        hood_w = 0.40 * W
        box(0.35, -hood_w / 2, xf - 0.38, hood_w, c["body"], zorder + 2, r=0.10)
        box(0.40, -hood_w * 0.68, 0.45 * L, hood_w * 1.36, c["body"], zorder + 2, r=0.08)
        box(0.55, -0.025, xf - 0.75, 0.05, c["body_dark"], zorder + 3, r=0.02)
        box(xf - 0.16, -hood_w * 0.35, 0.12, hood_w * 0.7, c["body_dark"], zorder + 3, r=0.04)
        # guardabarros traseros: mas cortos que la rueda y hacia dentro, para
        # que el neumatico asome por delante y por detras
        for sgn in (-1, 1):
            box(-rw_l * 0.36, sgn * (ry - 0.05) - 0.09, rw_l * 0.72, 0.18, c["body"],
                zorder + 2, r=0.07)
        # cabina y asiento
        cab_w = 0.62 * W
        box(-0.22, -cab_w / 2, 0.72, cab_w, c["cabin"], zorder + 3, r=0.10)
        box(-0.12, -cab_w * 0.3, 0.42, cab_w * 0.6, c["seat"], zorder + 4, r=0.06)
        box(0.36, -cab_w * 0.42, 0.08, cab_w * 0.84, c["body"], zorder + 4, r=0.03)

    def update(self, x: float, y: float, th: float, delta: float = 0.0) -> list:
        base = Affine2D().rotate(th).translate(x, y) + self.ax.transData
        for p, steer in self._parts:
            if steer is None:
                p.set_transform(base)
            else:
                p.set_transform(Affine2D().rotate_around(*steer, delta)
                                + Affine2D().rotate(th).translate(x, y) + self.ax.transData)
        return [p for p, _ in self._parts]


def split_by_direction(xy: np.ndarray, fwd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dos polilineas (adelante / atras) con NaN en los huecos, solapando el punto de cambio."""
    f, r = xy.astype(float).copy(), xy.astype(float).copy()
    f[~fwd] = np.nan
    r[fwd] = np.nan
    for i in np.flatnonzero(fwd[1:] != fwd[:-1]):
        f[i:i + 2] = r[i:i + 2] = xy[i:i + 2]
    return f, r
