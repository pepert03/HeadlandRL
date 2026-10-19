"""Muestreo de parametros por episodio segun la condicion experimental.

Condiciones del report (entrenamiento):
  A (fixed):    geometria y vehiculo fijos.
  B (geometry): varia cultivo; vehiculo fijo.
  C (dynamics): varia vehiculo (L, delta_max, tau_v, tau_delta); cultivo fijo.
  D (joint):    varia todo a la vez.
  E (privileged): como D, pero la politica recibe mu (se gestiona en el env
      con el flag `privileged=True`, no aqui).

Evaluacion (regimenes ID / OOD): pendiente de la fase de evaluacion.
Aqui solo se dejan definidos los nombres y un helper para desplazar rangos
fuera de distribucion, sin fijar valores finales.
"""
from __future__ import annotations

from dataclasses import replace
from enum import Enum
import numpy as np

from .params import CropParams, DomainRanges, EpisodeParams, VehicleParams


class DRMode(str, Enum):
    FIXED = "fixed"  # condicion A
    GEOMETRY = "geometry"  # condicion B
    DYNAMICS = "dynamics"  # condicion C
    JOINT = "joint"  # condicion D (y E, que anade obs privilegiada)


class EvalRegime(str, Enum):
    ID = "id"
    OOD_GEOMETRY = "ood_geometry"
    OOD_DYNAMICS = "ood_dynamics"
    OOD_JOINT = "ood_joint"
    FEASIBILITY_EDGE = "feasibility_edge"


def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def sample_crop(
    rng: np.random.Generator,
    ranges: DomainRanges,
    fixed: CropParams,
    randomize: bool,
) -> CropParams:
    if not randomize:
        return replace(fixed)
    n_lo, n_hi = ranges.n_lanes
    return replace(
        fixed,
        row_spacing=_uniform(rng, *ranges.row_spacing),
        headland_width=_uniform(rng, *ranges.headland_width),
        n_lanes=int(rng.integers(n_lo, n_hi + 1)),
    )


def sample_vehicle(
    rng: np.random.Generator,
    ranges: DomainRanges,
    fixed: VehicleParams,
    randomize: bool,
) -> VehicleParams:
    if not randomize:
        return replace(fixed)
    return replace(
        fixed,
        L=_uniform(rng, *ranges.L),
        delta_max_deg=_uniform(rng, *ranges.delta_max_deg),
        tau_v=_uniform(rng, *ranges.tau_v),
        tau_delta=_uniform(rng, *ranges.tau_delta),
    )


def sample_episode_params(
    mode: DRMode | str,
    rng: np.random.Generator,
    fixed_vehicle: VehicleParams,
    fixed_crop: CropParams,
    ranges: DomainRanges,
) -> EpisodeParams:
    """Muestrea los parametros de un episodio segun el modo DR."""
    mode = DRMode(mode)
    rand_geom = mode in (DRMode.GEOMETRY, DRMode.JOINT)
    rand_dyn = mode in (DRMode.DYNAMICS, DRMode.JOINT)
    vehicle = sample_vehicle(rng, ranges, fixed_vehicle, rand_dyn)
    crop = sample_crop(rng, ranges, fixed_crop, rand_geom)
    return EpisodeParams(vehicle=vehicle, crop=crop)


def ood_ranges(
    base: DomainRanges,
    axis: str,
    shift: float = 0.2,
) -> DomainRanges:
    """Helper provisional para evaluacion OOD (fase 7).

    Desplaza los rangos del eje indicado ("geometry" o "dynamics") una
    fraccion `shift` del ancho del rango hacia arriba. Es un placeholder:
    los valores finales de OOD se fijaran al definir las particiones de
    evaluacion (fase 4 del plan).
    """
    import dataclasses

    out = dataclasses.replace(base)
    if axis == "geometry":
        w_s = base.row_spacing[1] - base.row_spacing[0]
        w_h = base.headland_width[1] - base.headland_width[0]
        out.row_spacing = (base.row_spacing[1], base.row_spacing[1] + w_s * shift)
        out.headland_width = (
            base.headland_width[1],
            base.headland_width[1] + w_h * shift,
        )
    elif axis == "dynamics":
        w_tau = base.tau_v[1] - base.tau_v[0]
        out.tau_v = (base.tau_v[1], base.tau_v[1] + w_tau * shift)
    else:
        raise ValueError(f"axis desconocido: {axis}")
    return out
