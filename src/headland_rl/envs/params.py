"""Parametros del entorno: vehiculo, cultivo y rangos de domain randomization.

Rangos por defecto tomados del report (Propuesta revisada + "Modelo, parametros, etc."):
  - separacion entre calles: 1.8 - 3.2 m
  - anchura de cabecera:   3 - 10 m
  - calle destino:          1 - 4 calles de distancia
  - L (batalla):            1.4 - 2.0 m (Twizy real: 1.69 m)
  - delta_max:              20 - 35 deg (Twizy real: ~26 deg)
  - tau_v:                  0.1 - 0.5 s
  - tau_delta:              0.2 - 0.8 s

v_max, anchos de fila, ruidos iniciales y tolerancias NO estan en el report:
son decisiones de implementacion v0, marcadas como tales. v_max y la huella
salen del articulo del Twizy (Bengochea-Guevara et al., Smart Agric. Tech. 2025).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class VehicleParams:
    """Parametros del vehiculo (modelo bicicleta + actuacion de primer orden)."""

    L: float = 1.69  # batalla [m], Twizy real
    delta_max_deg: float = 26.0  # angulo max. direccion [deg], Twizy real ~26
    tau_v: float = 0.3  # cte. tiempo velocidad [s] (lazo cerrado completo)
    tau_delta: float = 0.5  # cte. tiempo direccion [s] (lazo cerrado completo)
    v_max: float = 0.83  # velocidad lineal maxima [m/s]: 3 km/h, tope de
    # operacion del Twizy en vinedo (Bengochea-Guevara et al. 2025)
    v_rev_max: float = 0.5  # marcha atras maxima [m/s] (decision v0; necesaria:
    # el nominal Twizy+H=6 no admite Dubins solo-avance, metrica 11 del report)
    gear_switch_time: float = 1.0  # [s] parado para cambiar adelante/atras
    # (decision v0: la marcha se cambia por CAN emulando los botones; a medir)
    # Huella (rectangulo) respecto al eje trasero, punto de referencia del
    # modelo bicicleta. Twizy: 2.32 x 1.19 m (chasis 1.23 m con ruedas de campo);
    # largo = rear_overhang + L + front_overhang. Voladizos repartidos a partes
    # iguales (decision v0, a medir).
    width: float = 1.23  # [m]
    front_overhang: float = 0.315  # [m] eje delantero -> parachoques
    rear_overhang: float = 0.315  # [m] eje trasero -> parachoques

    @property
    def delta_max(self) -> float:
        return math.radians(self.delta_max_deg)

    @property
    def r_min(self) -> float:
        """Radio minimo de giro R_min = L / tan(delta_max)."""
        return self.L / math.tan(self.delta_max)


@dataclass
class CropParams:
    """Parametros geometricos del cultivo / maniobra."""

    row_spacing: float = 2.5  # separacion entre calles [m]
    headland_width: float = 6.0  # anchura de cabecera [m]
    n_lanes: int = 1  # calles de distancia al destino (1-4)
    row_width: float = 0.5  # grosor de la fila de vinas [m] (decision v0)
    crop_depth: float = 5.0  # profundidad de cultivo simulada en -y [m] (v0)
    x_margin: float = 2.0  # margen lateral valido fuera de calles [m] (v0)
    y_margin_top: float = 1.0  # margen trasero valido mas alla de cabecera [m] (v0)


@dataclass
class InitNoise:
    """Ruido de la pose inicial (solo usado cuando la geometria se aleatoriza)."""

    x_std: float = 0.15  # [m] (decision v0)
    y_std: float = 0.15  # [m] (decision v0)
    theta_std_deg: float = 5.0  # [deg] (decision v0)


@dataclass
class DomainRanges:
    """Rangos de muestreo uniforme para DR (entrenamiento, dentro de distribucion)."""

    row_spacing: tuple[float, float] = (1.8, 3.2)
    headland_width: tuple[float, float] = (3.0, 10.0)
    n_lanes: tuple[int, int] = (1, 4)
    L: tuple[float, float] = (1.4, 2.0)
    delta_max_deg: tuple[float, float] = (20.0, 35.0)
    tau_v: tuple[float, float] = (0.1, 0.5)
    tau_delta: tuple[float, float] = (0.2, 0.8)


@dataclass
class EpisodeParams:
    """Parametros concretos de un episodio (ya muestreados)."""

    vehicle: VehicleParams = field(default_factory=VehicleParams)
    crop: CropParams = field(default_factory=CropParams)

    def as_privileged_vector(self) -> list[float]:
        """Vector mu de parametros reales (condicion E / oraculo)."""
        return [
            self.crop.row_spacing,
            self.crop.headland_width,
            float(self.crop.n_lanes),
            self.vehicle.L,
            self.vehicle.delta_max_deg,
            self.vehicle.tau_v,
            self.vehicle.tau_delta,
        ]

    @staticmethod
    def privileged_names() -> list[str]:
        return [
            "row_spacing",
            "headland_width",
            "n_lanes",
            "L",
            "delta_max_deg",
            "tau_v",
            "tau_delta",
        ]
