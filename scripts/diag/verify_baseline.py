"""Verifica el baseline clasico en el caso A (tolerancias estrictas)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from headland_rl.classical import DubinsPurePursuit
from headland_rl.envs.headland_env import HeadlandTurnEnv
from headland_rl.envs.geometry import goal_pose, start_pose

ok = 0
for ep in range(10):
    env = HeadlandTurnEnv(dr_mode="fixed", max_steps=500)
    obs, _ = env.reset(seed=ep)
    p = env.get_params()
    ctrl = DubinsPurePursuit()
    ctrl.reset_episode(p, start_pose(p), goal_pose(p))
    done = False
    while not done:
        x, y, th = env._state[:3]
        a = ctrl.act(float(x), float(y), float(th))
        _, _, term, trunc, info = env.step(a)
        done = term or trunc
    ok += int(info["success"])
    print(f"ep={ep} success={info['success']} coll={info['collided']} "
          f"dpos={info['dpos']:.2f} dth={info['dtheta_deg']:.1f} steps={info['steps']}")
print(f"EXITO: {ok}/10")
