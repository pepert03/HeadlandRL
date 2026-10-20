"""Visualiza una politica entrenada: animacion (GIF/MP4) o ventana en vivo.

Izquierda: escena vectorial en vista cenital (cabecera, calles, filas de vina)
con el vehiculo dibujado a escala, ruedas delanteras girando con la direccion
real, trayectoria (continua = adelante, discontinua = marcha atras), la
referencia del planificador en tenue y la pose objetivo como contorno.
Derecha: orden y respuesta de velocidad y de direccion, y la marcha engranada.

Uso:
  uv run python scripts/visualize_policy.py --model runs/A/seed_0/best_model.zip
  uv run python scripts/visualize_policy.py --model runs/A/seed_0/best_model.zip --show
  uv run python scripts/visualize_policy.py --model runs/A/seed_0/best_model.zip --out runs/A/demo.mp4
  uv run python scripts/visualize_policy.py --model random --dr-mode joint --episodes 3
Por defecto guarda junto al modelo: <modelo>_ep<k>.gif
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
import numpy as np

from headland_rl.config import build_env, load_config
from headland_rl.envs.geometry import start_pose
from headland_rl.viz import PALETTE, VehicleArtist, draw_scene, split_by_direction


def rollout(env, policy, seed: int) -> dict:
    """Ejecuta un episodio y guarda todo lo necesario para dibujarlo."""
    base = env.unwrapped  # estado y ordenes fisicas, con o sin NormalizedActions
    obs, _ = env.reset(seed=seed)
    rec = {"state": [base._state.copy()], "cmd": [np.zeros(2)], "gear": [base._gear]}
    done = False
    while not done:
        a = policy(obs)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
        rec["state"].append(base._state.copy())
        rec["cmd"].append(base._prev_action.copy())  # orden ya recortada, en m/s y rad
        rec["gear"].append(base._gear)
    rec = {k: np.array(v) for k, v in rec.items()}
    rec["info"], rec["params"], rec["goal"] = info, base.get_params(), base.get_goal()
    rec["ref"] = base._ref_override  # referencia del planificador (None si no hay)
    return rec


def outcome_text(info: dict) -> str:
    if info["success"]:
        return "EXITO"
    if info["collided"]:
        return f"COLISION ({info['collision_reason']})"
    return "TIEMPO AGOTADO"


def _style_series(ax) -> None:
    ax.set_facecolor("#fafafa")
    ax.grid(True, color="#e6e6e6", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999")
    ax.tick_params(colors="#555", labelsize=8)


def make_figure(rec: dict, dt: float, title: str):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    p, goal = rec["params"], rec["goal"]
    veh = p.vehicle
    t = np.arange(len(rec["state"])) * dt
    st, cmd = rec["state"], rec["cmd"]
    red, grey = PALETTE["path"], "#9a9a9a"

    fig = plt.figure(figsize=(10.5, 6.6), facecolor="white")
    gs = GridSpec(3, 2, width_ratios=[1.0, 1.15], figure=fig, hspace=0.5, wspace=0.08,
                  left=0.01, right=0.97, top=0.9, bottom=0.08)
    ax = fig.add_subplot(gs[:, 0])
    ax_v, ax_d, ax_g = (fig.add_subplot(gs[i, 1]) for i in range(3))
    fig.suptitle(title, fontsize=10, color="#333")

    # --- escena
    draw_scene(ax, p, float(goal[0]))
    if rec["ref"] is not None:
        ax.plot(rec["ref"][:, 0], rec["ref"][:, 1], color=PALETTE["ref"], lw=2.0,
                ls=(0, (1, 2)), zorder=4, solid_capstyle="round")
    VehicleArtist(ax, veh, zorder=4, alpha=0.9, ghost=True).update(*goal, 0.0)
    s0 = start_pose(p)
    VehicleArtist(ax, veh, zorder=4, alpha=0.25).update(*s0, 0.0)
    trail_f, = ax.plot([], [], color=red, lw=3.2, zorder=5, solid_capstyle="round")
    trail_r, = ax.plot([], [], color=red, lw=3.2, ls=(0, (2, 1.5)), zorder=5)
    car = VehicleArtist(ax, veh, zorder=6)
    status = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=9,
                     family="monospace", color="#222", zorder=20,
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888", alpha=0.9))

    # --- series temporales (completas, con un cursor que avanza)
    for a in (ax_v, ax_d, ax_g):
        _style_series(a)
    ax_v.plot(t, cmd[:, 0], color=grey, lw=1, ls="--", label="orden")
    ax_v.plot(t, st[:, 3], color=red, lw=1.8, label="real")
    ax_v.axhline(0, color="#777", lw=0.6)
    ax_v.set_title("velocidad [m/s]", fontsize=9, loc="left", color="#333")
    ax_v.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    ax_d.plot(t, np.degrees(cmd[:, 1]), color=grey, lw=1, ls="--", label="orden")
    ax_d.plot(t, np.degrees(st[:, 4]), color=red, lw=1.8, label="real")
    for lim in (-veh.delta_max_deg, veh.delta_max_deg):
        ax_d.axhline(lim, color=PALETTE["border"], lw=0.6, ls=":")
    ax_d.set_title("direccion [deg]", fontsize=9, loc="left", color="#333")
    ax_d.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    gear = rec["gear"]
    ax_g.fill_between(t, 0, gear, step="post", color=red, alpha=0.15, lw=0)
    ax_g.step(t, gear, where="post", color=red, lw=1.4)
    ax_g.set_yticks([-1, 1], ["atras", "adelante"])
    ax_g.set_ylim(-1.4, 1.4)
    ax_g.set_title("marcha", fontsize=9, loc="left", color="#333")
    ax_g.set_xlabel("t [s]", fontsize=8, color="#555")
    cursors = [a.axvline(0, color=PALETTE["border"], lw=1.0) for a in (ax_v, ax_d, ax_g)]

    fwd = st[:, 3] >= -0.01
    path_f, path_r = split_by_direction(st[:, :2], fwd)

    def draw(i: int):
        x, y, th, v, delta = st[i]
        trail_f.set_data(path_f[: i + 1, 0], path_f[: i + 1, 1])
        trail_r.set_data(path_r[: i + 1, 0], path_r[: i + 1, 1])
        parts = car.update(x, y, th, delta)
        for c in cursors:
            c.set_xdata([t[i], t[i]])
        last = i == len(st) - 1
        status.set_text(
            f"t {t[i]:5.1f} s   v {v:+.2f} m/s\n"
            f"delta {math.degrees(delta):+5.1f} deg   "
            f"{'adelante' if gear[i] > 0 else 'atras'}"
            + (f"\n{outcome_text(rec['info'])}" if last else "")
        )
        return [trail_f, trail_r, status, *parts, *cursors]

    return fig, draw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ruta .zip de SB3, o 'random'")
    ap.add_argument("--config", default="configs/train_A.yaml")
    ap.add_argument("--dr-mode", default=None, help="por defecto el de la config")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--every", type=int, default=1, help="dibuja 1 de cada N pasos")
    ap.add_argument("--speed", type=float, default=1.0, help="factor sobre tiempo real")
    ap.add_argument("--show", action="store_true", help="ventana en vivo en vez de fichero")
    ap.add_argument("--out", default=None, help=".gif o .mp4 (mp4 necesita ffmpeg)")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    cfg = load_config(args.config)
    overrides = {"dr_mode": args.dr_mode} if args.dr_mode else {}
    env = build_env(cfg, **overrides)
    base = env.unwrapped

    if args.model == "random":
        policy = lambda obs: env.action_space.sample()  # noqa: E731
        stem = Path("runs") / "random"
    else:
        from stable_baselines3 import PPO

        model = PPO.load(args.model)
        policy = lambda obs: model.predict(obs, deterministic=not args.stochastic)[0]  # noqa: E731
        stem = Path(args.model).with_suffix("")

    for ep in range(args.episodes):
        rec = rollout(env, policy, args.seed + ep)
        info, p = rec["info"], rec["params"]
        title = (f"{Path(args.model).name}  [{base.dr_mode.value}]  seed={args.seed + ep}\n"
                 f"H={p.crop.headland_width:.1f}m  calles={p.crop.n_lanes}x{p.crop.row_spacing:.2f}m  "
                 f"L={p.vehicle.L:.2f}m  dmax={p.vehicle.delta_max_deg:.0f}deg  "
                 f"tau_v={p.vehicle.tau_v:.2f}s  tau_d={p.vehicle.tau_delta:.2f}s")
        print(f"ep={ep} {outcome_text(info)} pasos={info['steps']} dpos={info['dpos']:.2f}m "
              f"dtheta={info['dtheta_deg']:.1f}deg cambios_marcha={info['gear_switches']}")

        fig, draw = make_figure(rec, base.dt, title)
        frames = list(range(0, len(rec["state"]), args.every))
        if frames[-1] != len(rec["state"]) - 1:
            frames.append(len(rec["state"]) - 1)
        frames += [frames[-1]] * 15  # pausa al final
        interval = 1000 * base.dt * args.every / args.speed
        anim = FuncAnimation(fig, draw, frames=frames, interval=interval, blit=False, repeat=False)
        if args.show:
            plt.show()
            continue
        out = Path(args.out) if args.out else Path(f"{stem}_ep{ep}.gif")
        if args.out and args.episodes > 1:
            out = out.with_name(f"{out.stem}_ep{ep}{out.suffix}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, round(1000 / interval))
        if out.suffix == ".gif":
            anim.save(str(out), writer=PillowWriter(fps=fps))
        else:
            anim.save(str(out), writer="ffmpeg", fps=fps)
        plt.close(fig)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
