# HeadlandRL

**Reinforcement learning for headland turns of an agricultural vehicle in a vineyard**, with geometric and dynamics domain randomization and an online-adaptation extension.

Master's thesis project. The reference vehicle is the Renault Twizy that the Centre for Automation and Robotics (CSIC-UPM) automated for vineyard inspection.

<p align="center">
  <img src="docs/media/paso_A_escena.gif" alt="Step A PPO policy performing a three-segment headland turn" width="300">
</p>

<p align="center"><sub>Step A PPO policy: it leaves the row, manoeuvres in the headland in three segments (forward, reverse, forward) and enters the next row. Solid line: forward; dashed line: reverse. Played at 2× speed.</sub></p>

---

## The problem

At the end of each row the vehicle has to leave the alley, manoeuvre in the headland and enter the next alley without touching the vines. The classical solution (Dubins curves plus Pure Pursuit) assumes a vehicle model that may not match the real one. A policy learned with RL can adapt better, but risks not generalizing to vineyards or vehicles it did not see during training.

The study systematically measures **how randomizing the crop geometry and the vehicle dynamics affects that generalization**, and whether a policy that **infers the domain from its recent history** recovers part of what is lost by not knowing it.

## Current result: step A

<p align="center">
  <img src="docs/media/paso_A.gif" alt="Scene and time series of speed, steering and gear during the manoeuvre" width="760">
</p>

| Metric | Value |
|---|---|
| Success rate, deterministic actions | **20/20** |
| Success rate, stochastic actions | **50/50** |
| Final position / heading error | 0.27 m / 3.1° (tolerance 0.30 m / 10°) |
| Duration and path length | 21.5 s, 11.2 m |
| Gear changes | 2 (three-segment manoeuvre) |
| Training | PPO, 3 M steps, ~15 min on CPU with 8 processes |

Nominal scenario: Twizy with a 1.69 m wheelbase and 26° maximum steering (3.47 m minimum turning radius), a 6 m headland and an adjacent target row 2.5 m away. With the vehicle's real footprint **there is no forward-only manoeuvre**, so the policy has to use reverse.

> Step A is the study's **lower baseline**: the policy has only ever seen this scenario. What makes it a baseline is evaluating it in the out-of-distribution regimes, which is still to be done.

## How it works

### Vehicle model

Kinematic bicycle model with first-order actuation, as in the proposal:

$$
\dot{x} = v\cos\theta,\quad \dot{y} = v\sin\theta,\quad \dot{\theta} = \frac{v}{L}\tan\delta,\quad
\dot{v} = \frac{v_{cmd}-v}{\tau_v},\quad \dot{\delta} = \frac{\delta_{cmd}-\delta}{\tau_\delta}
$$

Adjustments to the real vehicle (Bengochea-Guevara et al., 2025):

- **Speed** up to 0.83 m/s (3 km/h, the operating speed in the vineyard) and up to 0.5 m/s in reverse.
- **Footprint** of 2.32 × 1.23 m: collisions are checked against the whole vehicle, not a point.
- **Gear changes**: to reverse direction the vehicle brakes to a stop and waits 1 s, because the Twizy's gear is changed over CAN by emulating its buttons.

The policy acts at 10 Hz. It observes the pose, speed, steering angle, error to the goal and engaged gear, and outputs speed and steering commands.

### Experimental conditions

<p align="center">
  <img src="docs/media/escenarios_dr.png" alt="Four randomized scenarios with different headland, row spacing and vehicle" width="820">
</p>

<p align="center"><sub>Condition D scenarios: headland width, row spacing, distance to the target row, and the vehicle's wheelbase and maximum steering all vary.</sub></p>

| Step | What varies during training | Role | Status |
|---|---|---|---|
| **A** | Nothing (nominal scenario) | Lower baseline | ✅ trains; OOD evaluation pending |
| **B** | Crop geometry | Isolates geometric uncertainty | ⏳ |
| **C** | Vehicle dynamics (L, δmax, τv, τδ) | Isolates dynamics uncertainty | ⏳ |
| **D** | Both | Realistic condition; interaction analysis | ⏳ |
| **E** | Both, and the policy receives the true parameters | Oracle (upper baseline) | ⏳ |
| **F** | Both, and the policy infers them from its history | Online adaptation (extension) | ⏳ |

All of them will be compared with a classical, non-learning method in five evaluation regimes: in-distribution, out-of-distribution in geometry, in dynamics and in both, and the feasibility boundary.

### How the manoeuvre is learned

With random exploration the vehicle always crashes and never discovers a three-segment manoeuvre with gear changes. What makes it learnable:

1. **Expert reference.** A Hybrid A* planner that accounts for the real footprint computes a valid manoeuvre (`scripts/plan_reference.py`).
2. **Backplay with an adaptive curriculum.** The vehicle first spawns near the end of the reference, and the spawn point moves back **only once the policy succeeds from there 60% of the time**. With a fixed schedule, the reverse segment went by without being learned.
3. **Real conditions from the first step.** Speed, goal tolerance and gear-change time are not relaxed during training. When they were, the policy learned conditions that did not match the evaluation.
4. **Progress-along-the-reference reward.** Euclidean distance to the goal penalizes the reverse segment, which moves away from it. Progress only counts in the segment's direction of travel, and every gear change is penalized, so oscillating back and forth does not pay off.
5. **Exploration compatible with gear changes.** Normalized actions and small, time-correlated initial noise (gSDE). With the default noise, the command flipped sign almost every step and the policy ended up standing still.

## Usage

Requires [uv](https://docs.astral.sh/uv/). **Do not put the project in OneDrive**: the university's cloud antivirus quarantines package files inside `.venv`.

```bash
uv sync --group dev --group train
uv run pytest tests/ -q
```

Step A:

```bash
uv run python scripts/plan_reference.py      # only if data/references/reference_A.npy is missing (~1 min)
uv run python scripts/train.py --config configs/train_A.yaml --seed 0
uv run python scripts/evaluate.py --model runs/A/seed_0/best_model.zip --episodes 20
```

Visualization:

```bash
uv run python scripts/visualize_policy.py --model runs/A/seed_0/best_model.zip          # GIF next to the model
uv run python scripts/visualize_policy.py --model runs/A/seed_0/best_model.zip --show   # live window
uv run python scripts/visualize_policy.py --model random --dr-mode joint --episodes 3   # random policy
```

Useful options: `--scene-only` (scene only), `--every 2 --speed 2` (lighter and faster), `--out demo.mp4` (requires ffmpeg).

## Layout

```
configs/
  env_default.yaml         # ranges from the proposal and implementation decisions
  train_A.yaml             # step A: PPO, curriculum, reward
data/references/           # expert references for backplay
docs/media/                # GIFs and images for this README
src/headland_rl/
  config.py                # build_env(cfg): the same environment for training, evaluation and visualization
  viz.py                   # vector drawing of the scene and the vehicle
  envs/
    headland_env.py        # HeadlandTurnEnv (Gymnasium): dynamics, reward, curriculum
    wrappers.py            # actions normalized to [-1, 1]
    params.py              # VehicleParams / CropParams / DomainRanges
    randomization.py       # randomization modes A-D and evaluation regimes
    geometry.py            # poses, footprint collisions, feasibility
    dubins.py              # Dubins paths
  classical/pure_pursuit.py  # classical Dubins + Pure Pursuit method
scripts/
  train.py                 # PPO (SB3) with adaptive curriculum and SubprocVecEnv
  evaluate.py              # main metrics -> CSV + JSON
  plan_reference.py        # Hybrid A* with the vehicle footprint
  visualize_policy.py      # animation (GIF/MP4/window)
  diag/                    # one-off probes and checks
tests/
report/                    # thesis proposal (local only, not versioned)
runs/                      # training outputs (not versioned)
```

## Roadmap

- [x] Parameterized environment with modes A–D and privileged observation E
- [x] Simulator tuned to the Twizy: speed, footprint and gear changes
- [x] Step A: PPO solves the nominal scenario
- [ ] Evaluation by regime (ID, geometric, dynamics and joint OOD, feasibility boundary)
- [ ] Per-episode backplay reference (Hybrid A* or Reeds–Shepp), needed for B–F
- [ ] Classical baseline with reverse (planner + tracker); the current one only drives forward
- [ ] Steps B–E with several seeds and statistical tests
- [ ] Step F: memory-based policy and RMA-style encoder
- [ ] Smoother driving (steering currently goes to full lock and speed is jerky) and deployment on the Twizy (ROS + ONNX)

## Implementation decisions

Not specified in the proposal; marked in the code:

- `v_rev_max = 0.5 m/s`, `gear_switch_time = 1 s`, 0.315 m front and rear overhangs.
- `row_width = 0.5 m`, `crop_depth = 5 m` and valid-area margins.
- Initial pose noise `(0.15 m, 0.15 m, 5°)`, only when geometry is randomized.
- Step A: `dt = 0.1 s`, `max_steps = 500` (50 s) and success tolerances of 0.30 m and 10°. The environment defaults are `dt = 0.05 s` and `max_steps = 1000`.
- Fixed action space, clipped to each episode's limits so that it does not change under randomization.

**Open issue:** with the Twizy's footprint, the narrowest rows in the proposal's range (1.8 m) leave almost no clearance, and ~10% of B and D episodes already start in collision. We need to decide whether to raise the minimum, also randomize the vehicle width, or discard infeasible episodes.

## References

- Bengochea-Guevara, J. M. et al. *Drive-by-wire automation and field validation of a commercial electric vehicle for autonomous crop monitoring*. Smart Agricultural Technology 12 (2025) 101530.
- Pindl, L. et al. *Reinforcement Learning for Autonomous Headland Turns*. ICINCO 2024.
- Salimans, T. and Chen, R. *Learning Montezuma's Revenge from a Single Demonstration*. 2018 (backplay).
- Kumar, A. et al. *RMA: Rapid Motor Adaptation for Legged Robots*. RSS 2021.
- Stable-Baselines3: https://stable-baselines3.readthedocs.io
