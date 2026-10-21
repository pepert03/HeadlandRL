"""Entrenamiento PPO para una condicion experimental (paso A: dr_mode=fixed).

Uso:
  uv run python scripts/train.py --config configs/train_A.yaml --seed 0
  uv run python scripts/train.py --config configs/train_A.yaml --seed 1 --timesteps 50000
Guarda en runs/<condition>/seed_<seed>/: ppo_final.zip, best_model.zip, tensorboard.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import headland_rl  # noqa: F401 (registra HeadlandTurn-v0 en Gymnasium)
from headland_rl.config import build_env, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_A.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None, help="pisa n_envs de la config")
    ap.add_argument("--vec-env", choices=["dummy", "subproc"], default=None,
                    help="pisa vec_env de la config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    total = args.timesteps or int(cfg.get("total_timesteps", 300_000))
    n_envs = args.n_envs or int(cfg.get("n_envs", 4))
    vec_env = args.vec_env or cfg.get("vec_env", "dummy")
    outdir = Path("runs") / str(cfg.get("condition", "A")) / f"seed_{args.seed}"
    (outdir / "tb").mkdir(parents=True, exist_ok=True)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        EvalCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    # Una MLP de 64x64 va mas rapida con 1 hilo: con varios, el reparto entre
    # hilos cuesta mas que el calculo y compite con los procesos de SubprocVecEnv
    torch.set_num_threads(int(cfg.get("torch_threads", 1)))

    class CurriculumCallback(BaseCallback):
        """Curriculum del entorno de rollout (tolerancias, velocidad, marcha, backplay).

        Solo toca el entorno de entrenamiento; la evaluacion (EvalCallback)
        usa siempre los valores estrictos, que son la metrica real. Se
        actualiza una vez por rollout (no por paso): con SubprocVecEnv cada
        llamada es un viaje de ida y vuelta a todos los procesos.

        mode=linear: frac = pasos / steps (avanza aprenda o no).
        mode=adaptive: frac sube `step` solo cuando el exito de los episodios
        que arrancan en la frontera del backplay supera `threshold` (con al
        menos `min_episodes`); si no, la politica sigue practicando ese tramo.
        `phys_frac` fija las condiciones fisicas (tolerancias, velocidad, marcha);
        en adaptive por defecto 1.0 = las reales, para que lo aprendido en cada
        tramo valga tal cual en la evaluacion. En linear siguen a frac.
        """

        def __init__(self, cc: dict, verbose: int = 0):
            super().__init__(verbose)
            self.mode = cc.get("mode", "linear")
            self.steps = max(int(cc.get("steps", 300_000)), 1)
            self.step = float(cc.get("step", 0.025))
            self.threshold = float(cc.get("threshold", 0.6))
            self.min_episodes = int(cc.get("min_episodes", 100))
            self.phys_frac = cc.get("phys_frac", 1.0 if self.mode == "adaptive" else None)
            self.frac, self.n, self.ok = 0.0, 0, 0

        def _on_rollout_start(self) -> None:
            if self.mode == "linear":
                self.frac = self.num_timesteps / self.steps
            elif self.n >= self.min_episodes:
                rate = self.ok / self.n
                self.logger.record("curriculum/frontier_success", rate)
                if rate >= self.threshold:
                    self.frac = min(1.0, self.frac + self.step)
                self.n, self.ok = 0, 0
            self.logger.record("curriculum/frac", self.frac)
            self.training_env.env_method("set_curriculum", self.frac, self.phys_frac)

        def _on_step(self) -> bool:
            if self.mode == "adaptive":
                for done, info in zip(self.locals["dones"], self.locals["infos"]):
                    if done and info.get("at_frontier"):
                        self.n += 1
                        self.ok += int(info.get("success", False))
            return True

    # subproc: un proceso por entorno (usa varios nucleos); dummy: todos en este proceso
    vec_cls = SubprocVecEnv if vec_env == "subproc" else DummyVecEnv
    train_env = vec_cls(
        [lambda r=r: Monitor(build_env(cfg, args.seed + 1000 + r)) for r in range(n_envs)]
    )
    eval_env = DummyVecEnv([lambda: Monitor(build_env(cfg, 10_000 + args.seed))])
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(outdir),
        log_path=str(outdir),
        eval_freq=int(cfg.get("eval_freq", 10_000)),
        n_eval_episodes=int(cfg.get("n_eval_episodes", 10)),
        deterministic=True,
    )
    callbacks = CallbackList([
        CurriculumCallback(cfg.get("curriculum", {"steps": cfg.get("curriculum_steps", 300_000)})),
        eval_cb,
    ])

    pc = dict(cfg.get("ppo", {}))
    policy_kwargs = {"net_arch": pc.pop("net_arch", [64, 64])}
    if "log_std_init" in pc:
        policy_kwargs["log_std_init"] = pc.pop("log_std_init")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(outdir / "tb"),
        policy_kwargs=policy_kwargs,
        **pc,
    )
    # Prior de avance (fase 4): la media inicial de v_cmd se desplaza +0.5
    # (en m/s, o en unidades normalizadas si normalize_actions).
    # Sin esto, las colisiones tempranas empujan la media a v<0 (recortada a 0
    # por el espacio de accion) y la politica queda aparcada para siempre en
    # un optimo local del que PPO no escapa (gradiente nulo en el clip).
    with torch.no_grad():
        model.policy.action_net.bias[0] += 0.5
    print(f"[train] condicion={cfg.get('condition')} dr_mode={cfg.get('dr_mode')} "
          f"seed={args.seed} timesteps={total} n_envs={n_envs} vec_env={vec_env} -> {outdir}")
    model.learn(total_timesteps=total, callback=callbacks)
    model.save(str(outdir / "ppo_final"))
    train_env.close()
    eval_env.close()
    print(f"[train] OK: {outdir / 'ppo_final.zip'} (mejor: {outdir / 'best_model.zip'})")


if __name__ == "__main__":
    main()
