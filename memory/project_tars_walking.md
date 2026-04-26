---
name: TARS Walking Project
description: MuJoCo quadruped robot RL simulation using PPO, with known phase-1 gait bug and hardcoded Windows paths in tests
type: project
---

## Repo: /home/awebb/Research/TARS_Walking (cloned from github.com/CPP-AdvancedComputing/TARS_Walking)

### Architecture
- **TARSEnv** (`tars_env.py`, 2091 lines): gymnasium env wrapping MuJoCo sim
- **Training**: `train.py` with PPO (stable-baselines3), `training_helpers.py` has CurriculumCallback + BestWalkCallback
- **Model**: `tars_mjcf.xml` (MJCF) or `robot_mujoco.urdf` (URDF fallback); `tars_model.py` resolves path
- **Gait reference**: `tars_gait_reference.py` — phase poses, DISPLAY_TO_MODEL_LEG mapping
- **Remote runner**: `tide_tars.py` for TIDE/JupyterHub

### Key Constants
- ACTION_DIM=4 (swing_angle, swing_length, plant_angle, plant_length)
- Obs space: 46D (qpos[19] + qvel[18] + xpos[3] + xquat[4] + clock[2])
- FRAME_SKIP=5, timestep=0.002s → 10ms/action
- PHASE_STEPS=30 → 0.3s/phase, 0.6s full cycle
- MAX_EPISODE_STEPS=2000 (env var TARS_MAX_EPISODE_STEPS)
- SUPPORT_PAIR=(0,2), SWING_PAIR=(1,3)
- HIP_STANDING_SIGNS=(-1,1,-1,1) — compensates for physical leg asymmetry

### Gait: Parallel Crutch
- Phase 0: support=(0,2), swing=(1,3) — healthy phase
- Phase 1: support=(1,3), swing=(0,2) — buggy phase
- Phase switches when next plant feet approach ground (PHASE_SWITCH_GROUND_Z=0.03m)
- Phase switch timeout: 60 steps (resets always start at phase 0)

### Known Bugs

**1. Phase 1 support failure (PRIMARY BUG)**
- Desired contacts: [0,1,0,1], actual: [1,0,0,1]
- Leg 1 loses ground contact while leg 0 re-contacts
- Root cause candidate: structural asymmetry in tars_mjcf.xml
  - servo_l0: pos z = -0.0375 (negative)
  - servo_l1: pos z = +0.0375 (POSITIVE — sign flip)
  - fixed_carriage_l0: pos z = +0.025693; fixed_carriage_l1: pos z = -0.018156
- HIP_STANDING_SIGNS=(-1,1,-1,1) partially compensates, but phase-1 dynamic support still fails
- reset() always starts at phase=0 with comment: "Default to the healthier support pattern while phase-1 transition dynamics are still being debugged"

**2. Hardcoded Windows paths in test files (ALL test files)**
- test_gait_phase.py: `URDF = r"C:\Users\anike\tars-urdf\tars_mjcf.xml"`
- test_survive.py: same
- test_sanity.py: same
- test_env.py: same hardcoded path
- All tests will fail on Linux without path fix → replace with DEFAULT_MODEL_PATH_STR from tars_model

**3. Reward function fully commented out except progress_reward**
- Only `progress_reward` is active in step() (line 2059)
- All gait shaping, contact quality, swing incentives, penalties commented out
- This is the "foundation" profile but makes learning the gait essentially unsupervised

### Diagnostic Files
- `diagnose_phase_support.py` — phase support analysis
- `audit_leg1_geometry.py` — leg geometry audit
- `debug_gait_inspector.py` — gait inspection

### Training Results (2026-04-11)
- ep_rew_mean ~277, progress_gate ~0.826
- Phase 0 contacts coherent, phase 1 still fails

### Fixes Applied (2026-04-12)
- Tripod gait: SUPPORT_PAIR=(0,3), SWING_PAIR=BOUND_PAIR=(1,2); phase 0 plants L0+L3, phase 1 plants L1+L2
- ACTION_DIM 4→6: L0_angle/length, L12_angle/length (shared bound pair), L3_angle/length
- Reward fully restored: progress*gate + gait_reference + shaping + swing_lift + penalties
- PLANT_UNLOAD_PENALTY_SCALE now implemented; PHASE_STALL_PENALTY 0.1→2.0
- Observation 46→41D: absolute XY removed
- Reset randomizes start phase 50/50
- Curriculum reversed: starts at 0.35 authority and expands to 1.0
- Per-leg rod targets for bound pair; pair feedback only locks BOUND_PAIR
- _shaping_terms double-count removed; REFERENCE_HIP_AMPLITUDE 0.15→0.30
- All test Windows paths fixed; 20 passing, 2 xfailed

### Training Status
- 1M-step PPO run submitted to TIDE (GPU 0, A100) 2026-04-12
- submit_train.py: main training submission script
- visualize.py: headless renderer → MP4 + --diag mode for contact analysis

### Next Steps
- Wait for 1M-step run to complete and download policy
- Run visualize.py --policy tide_tars_policy.zip to inspect learned gait
- If phase 1 still weak: check RESET_PLANT_POSE for legs 1,2 geometry
- Consider longer run (2-5M steps) once basic gait is learned
