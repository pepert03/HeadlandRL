# HeadlandRL

Estudio factorial de DR geometrica vs dinamica en headland turns (viñedo),
segun `report/Propuesta_TFM_cambio_de_calle.pdf` (propuesta revisada) y
`report/Modelo, parámetros, etc.pdf`.

## Estado actual

| Pieza | Estado |
|---|---|
| Entorno `HeadlandTurn-v0` (modos A-D, flag privilegiado E) | hecho, con tests |
| Recompensa | v3 (progreso euclideo + terminales), ver `headland_env.py` |
| Paso A (PPO sin DR) | 100/100 exitos en ID (episodio determinista) |
| Pasos B-E | pendiente: el backplay usa una referencia fija valida solo para A |
| Paso F (adaptacion en linea) | pendiente |
| Baseline clasico | `DubinsPurePursuit` (solo avance: no resuelve el caso nominal) |
| Evaluacion por regimenes ID/OOD/frontera | pendiente (`ood_ranges` es un placeholder) |

## Modelo

Bicicleta + actuacion de primer orden (`src/headland_rl/envs/headland_env.py`):
`x_dot=v cos th`, `y_dot=v sin th`, `th_dot=(v/L) tan delta`,
`v_dot=(v_cmd-v)/tau_v`, `delta_dot=(delta_cmd-delta)/tau_delta`,
con `|delta| <= delta_max` y `-v_rev_max <= v <= v_max`.

Diferencia con el report: se admite **marcha atras** (`v_cmd in [-1, 2]` m/s).
El caso nominal (Twizy Rmin=3.47m, H=6m) no admite maniobra solo-avance
(`scripts/diag/check_dubins_feasibility.py`).

## Estructura

```
configs/
  env_default.yaml         # config por defecto (rangos del report + decisiones v0)
  train_A.yaml             # paso A
data/
  references/              # referencias expertas para backplay (versionadas)
  demos/                   # pares (obs, act) para BC
src/headland_rl/
  __init__.py              # registra "HeadlandTurn-v0" en Gymnasium
  envs/
    headland_env.py        # HeadlandTurnEnv
    params.py              # VehicleParams / CropParams / DomainRanges
    randomization.py       # DRMode (A-D), EvalRegime
    geometry.py            # start/goal, colisiones, factibilidad
    dubins.py              # longitud y muestreo de caminos Dubins
  classical/
    pure_pursuit.py        # baseline Dubins + Pure Pursuit
scripts/
  train.py                 # PPO por condicion
  evaluate.py              # metricas principales -> CSV + JSON
  search_reference.py      # busca la referencia de A -> data/references/reference_A.npy
  expert_3point.py         # experto guionado alternativo -> reference_A_expert3p.npy
  run_random_agent.py, view_env.py
  diag/                    # sondas y comprobaciones puntuales
tests/
report/                    # propuesta (LaTeX), figuras y documentos del TFM
runs/                      # salidas de entrenamiento (no versionado)
```

## Uso (con uv)

```bash
uv sync --group dev --group train
uv run pytest tests/ -q
uv run python scripts/run_random_agent.py --episodes 5   # imagen en runs/
uv run python scripts/view_env.py --episodes 3           # ventana en vivo
```

### Paso A

```bash
uv run python scripts/search_reference.py   # solo si falta data/references/reference_A.npy
uv run python scripts/train.py --config configs/train_A.yaml --seed 0
uv run python scripts/evaluate.py --model runs/A/seed_0/ppo_final.zip --episodes 100
```
Resultado (ID, determinista, 100 eps): `success_rate=1.0`, `collision_rate=0.0`,
`dpos=0.26m`, `dtheta=6.9°`, `246 steps`, `path=15.0m`. En modo `fixed` no hay
ruido inicial, asi que los 100 episodios son la misma trayectoria.

Hallazgos metodologicos (fase 4):
- Recompensa v3: progreso euclideo + `success+100` / colision `-10` con coste
  de oportunidad (chocar pronto no ahorra tiempo) + bonus de zona.
- Curriculum triple (tolerancias 2m/90°->estrictas, velocidad 0.6->2m/s,
  **backplay** sobre la referencia experta con 80% frontera + 20% ensayo).
- Sin backplay, PPO no descubre la maniobra (exploracion aleatoria = 100%
  colisiones; recompensa euclidea ingenua -> optimo local "aparcado").

## Decisiones v0 (no estan en el report, marcadas en codigo)

- `v_max=2.0 m/s`, `v_rev_max=1.0 m/s`, `row_width=0.5 m`, `crop_depth=5 m`, margenes de zona valida.
- Ruido inicial `(0.15 m, 0.15 m, 5 deg)` solo cuando la geometria se aleatoriza.
- `dt=0.05 s`, `max_steps=400` (500 en `train_A.yaml`), tolerancias de exito `0.30 m / 10 deg`.
- Espacio de accion fijo `[-1,2] m/s x [-35,+35] deg`, recortado a los limites
  del episodio (para que el espacio no cambie con DR).
- Colisiones: en cultivo solo pasillos libres; en cabecera todo libre.

## Pendientes conocidos

- Referencia de backplay por episodio (Reeds-Shepp) para B-F; tambien como baseline clasico.
- `evaluate.py` no lee la config de entrenamiento ni soporta `privileged`.
- Accion/criterio de "abortar" para medir el reconocimiento de casos inviables.
