import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from headland_rl.envs.headland_env import HeadlandTurnEnv
from headland_rl.envs.randomization import DRMode


def test_reset_step_shapes():
    env = HeadlandTurnEnv(dr_mode="fixed")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (10,)
    for _ in range(5):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        assert obs.shape == (10,)
        assert isinstance(rew, float)
        assert "success" in info and "collided" in info
        if term or trunc:
            break


def test_privileged_obs_dim():
    env = HeadlandTurnEnv(dr_mode="joint", privileged=True)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (17,)
    assert ((obs[10:] >= 0.0) & (obs[10:] <= 1.0)).all()


def test_dr_modes_change_params():
    seen = set()
    env = HeadlandTurnEnv(dr_mode="joint")
    for s in range(10):
        env.reset(seed=s)
        p = env.get_params()
        seen.add((round(p.crop.row_spacing, 3), round(p.vehicle.L, 3)))
    assert len(seen) > 1, "joint deberia variar params entre episodios"

    env_f = HeadlandTurnEnv(dr_mode="fixed")
    env_f.reset(seed=0)
    p0 = env_f.get_params()
    env_f.reset(seed=999)
    p1 = env_f.get_params()
    assert p0.crop.row_spacing == p1.crop.row_spacing
    assert p0.vehicle.L == p1.vehicle.L


def test_action_clipped_to_episode_delta_max():
    env = HeadlandTurnEnv(dr_mode="fixed")
    env.reset(seed=0)
    d_abs = math.radians(35.0)
    # accion al limite del espacio debe recortarse al delta_max del episodio (26 deg Twizy)
    obs, *_ = env.step([2.0, d_abs])
    assert abs(env._state[4]) <= env.get_params().vehicle.delta_max + 1e-9


def test_collision_in_vines():
    env = HeadlandTurnEnv(dr_mode="fixed")
    env.reset(seed=0)
    # Teletransporte manual a mitad de una fila de vinas (entre calles)
    p = env.get_params()
    env._state[0] = p.crop.row_spacing / 2.0  # entre calle 0 y calle 1
    env._state[1] = -1.0
    _, _, term, _, info = env.step([0.0, 0.0])
    assert info["collided"] and term
