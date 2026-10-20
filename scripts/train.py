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
    args = ap.parse_args()

    cfg = load_config(args.config)
    total = args.timesteps or int(cfg.get("total_timesteps", 300_000))
    n_envs = int(cfg.get("n_envs", 4))
    outdir = Path("runs") / str(cfg.get("condition", "A")) / f"seed_{args.seed}"
    (outdir / "tb").mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        EvalCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    class CurriculumCallback(BaseCallback):
        """Endurece pos_tol/ang_tol del env de rollout de 0 a `curr_steps`.

        Solo toca el entorno de entrenamiento; la evaluacion (EvalCallback)
        usa siempre las tolerancias estrictas, que son la metrica real.
        """

        def __init__(self, curr_steps: int, verbose: int = 0):
            super().__init__(verbose)
            self.curr_steps = max(int(curr_steps), 1)

        def _on_step(self) -> bool:
            frac = self.num_timesteps / self.curr_steps
            self.training_env.env_method("set_curriculum", frac)
            return True

    train_env = DummyVecEnv(
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
        CurriculumCallback(int(cfg.get("curriculum_steps", 300_000))),
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
    import torch

    with torch.no_grad():
        model.policy.action_net.bias[0] += 0.5
    print(f"[train] condicion={cfg.get('condition')} dr_mode={cfg.get('dr_mode')} "
          f"seed={args.seed} timesteps={total} -> {outdir}")
    model.learn(total_timesteps=total, callback=callbacks)
    model.save(str(outdir / "ppo_final"))
    train_env.close()
    eval_env.close()
    print(f"[train] OK: {outdir / 'ppo_final.zip'} (mejor: {outdir / 'best_model.zip'})")


if __name__ == "__main__":
    main()
