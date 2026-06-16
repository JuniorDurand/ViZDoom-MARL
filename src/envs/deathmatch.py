"""
Deathmatch environment for ViZDoom.

Supports two modes:
  - "single": one player against built-in monsters (fast, for pretraining)
  - "networked": multiplayer FFA via sockets (slow, for competitive experiments)

Reward shaping follows the CIG competition standard with 9 configurable
components. Agents respawn on death; episodes end by timeout.

Compatible with both value-based (DQN/DRQN) and policy-gradient (PPO)
trainers — exposes a flat per-agent interface:
    reset()  → obs (uint8 numpy array, shape: frame_stack*3, H, W)
    step(action_idx) → obs, reward, done, info
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import vizdoom as vzd


# ---------------------------------------------------------------------------
# Action space: 19 composite actions covering movement, strafing, turning,
# attacking, sprinting, and weapon switching.  Matches the CIG competition
# setup so results are directly comparable.
# ---------------------------------------------------------------------------

BUTTONS = [
    vzd.Button.ATTACK,
    vzd.Button.SPEED,
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.SELECT_NEXT_WEAPON,
    # vzd.Button.RELOAD
]

ACTIONS = [
    [0, 0, 1, 0, 0, 0, 0, 0, 0],  # forward
    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # backward
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # turn left
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # turn right
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # strafe left
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # strafe right
    [0, 0, 1, 0, 1, 0, 0, 0, 0],  # forward + turn left
    [0, 0, 1, 0, 0, 1, 0, 0, 0],  # forward + turn right
    [0, 1, 1, 0, 0, 0, 0, 0, 0],  # sprint forward
    [0, 1, 1, 0, 1, 0, 0, 0, 0],  # sprint forward + turn left
    [0, 1, 1, 0, 0, 1, 0, 0, 0],  # sprint forward + turn right
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # attack
    [1, 0, 1, 0, 0, 0, 0, 0, 0],  # attack + forward
    [1, 0, 0, 1, 0, 0, 0, 0, 0],  # attack + backward
    [1, 0, 0, 0, 0, 0, 1, 0, 0],  # attack + strafe left
    [1, 0, 0, 0, 0, 0, 0, 1, 0],  # attack + strafe right
    [1, 0, 0, 0, 1, 0, 0, 0, 0],  # attack + turn left
    [1, 0, 0, 0, 0, 1, 0, 0, 0],  # attack + turn right
    # [0, 0, 0, 0, 0, 0, 0, 0, 1],  # next weapon
]

# Game variables tracked for reward shaping and metrics
GAME_VARS = [
    vzd.GameVariable.FRAGCOUNT,
    vzd.GameVariable.KILLCOUNT, 
    vzd.GameVariable.HEALTH,
    vzd.GameVariable.ARMOR,
    vzd.GameVariable.DAMAGECOUNT,
    vzd.GameVariable.DAMAGE_TAKEN,
    vzd.GameVariable.HITCOUNT,
    vzd.GameVariable.ITEMCOUNT,
    vzd.GameVariable.SELECTED_WEAPON_AMMO,
    vzd.GameVariable.DEATHCOUNT,
]


@dataclass
class RewardConfig:
    """Weights for each reward component. Defaults match CIG competition."""
    frag:         float =  1.0
    damage:       float =  0.01
    damage_taken: float = -0.005
    hit:          float =  0.02
    item:         float =  0.05
    health:       float =  0.001
    ammo:         float =  0.005
    death:        float =  0.5
    living:       float =  0.0005
    miss:         float =  0.00
    clip_min:     float = -3.0
    clip_max:     float =  5.0


@dataclass
class EnvConfig:
    """Environment configuration."""
    scenario:        str   = "cig"          # "cig" or "deathmatch"
    mode:            str   = "single"       # "single" or "networked"
    rank:            int   = 0
    n_players:       int   = 1
    port:            int   = 5029
    episode_minutes: float = 4.0
    frame_skip:      int   = 4
    frame_stack:     int   = 4
    obs_resolution:  tuple = (84, 84)
    screen_resolution: vzd.ScreenResolution = vzd.ScreenResolution.RES_640X480
    doom_skill:      int   = 4
    record:          bool  = False
    max_rec_frames:  int   = 4000
    reward:          RewardConfig = field(default_factory=RewardConfig)

    @property
    def in_channels(self) -> int:
        """Number of input channels for the network (frame_stack * 3 for RGB)."""
        return self.frame_stack * 3


class DeathmatchEnv:
    """
    Per-agent ViZDoom deathmatch environment.

    Each agent gets its own instance. In networked mode, instances connect
    via sockets (rank 0 = host, others = clients). In single mode, each
    instance runs independently against built-in monsters.

    IMPORTANT: networked mode requires multiprocessing (one process per agent).
    ViZDoom is not thread-safe with multiple game instances. The trainer must
    spawn each agent in a separate process.

    Observations are RGB, shape (frame_stack * 3, H, W), dtype uint8.
    """

    def __init__(self, config: Optional[EnvConfig] = None, auto_init: bool = True, **kwargs):
        if config is None:
            config = EnvConfig(**kwargs)
        self.cfg = config

        self.n_actions = len(ACTIONS)
        self.in_channels = config.in_channels
        self.actions = ACTIONS
        self.frames = deque(maxlen=config.frame_stack)
        self.recorded_frames = []
        self._prev_vars = {v: 0.0 for v in GAME_VARS}

        # episode-level metrics tracking
        self._episode_steps = 0
        self._steps_since_death = 0
        self._attack_steps = 0

        self.game = self._create_game()
        if auto_init:
            self.game.init()

    def _create_game(self) -> vzd.DoomGame:
        cfg = self.cfg
        game = vzd.DoomGame()

        scenario_path = Path(vzd.scenarios_path) / f"{cfg.scenario}.cfg"
        if not scenario_path.is_file():
            raise FileNotFoundError(
                f"Scenario not found: {scenario_path}\n"
                f"Available: {[f.stem for f in Path(vzd.scenarios_path).glob('*.cfg')]}"
            )
        game.load_config(str(scenario_path))

        game.set_available_buttons(BUTTONS)
        game.set_available_game_variables(GAME_VARS)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_window_visible(False)
        game.set_console_enabled(False)
        game.set_doom_skill(cfg.doom_skill)

        if cfg.record:
            game.set_screen_resolution(cfg.screen_resolution)
        else:
            game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)

        if cfg.mode == "networked" and cfg.n_players > 1:
            game.set_mode(vzd.Mode.PLAYER)

            if cfg.rank == 0:
                game.add_game_args(
                    f"-host {cfg.n_players} -port {cfg.port} "
                    f"-netmode 0 -deathmatch "
                    f"+timelimit {cfg.episode_minutes} "
                    "+sv_forcerespawn 1 +sv_noautoaim 1 "
                    "+sv_respawnprotect 1 +sv_spawnfarthest 1 "
                    "+sv_nocrouch 1 +viz_respawn_delay 2 "
                    "+viz_nocheat 1 +sv_losefrag 1"
                )
            else:
                game.add_game_args(
                    f"-join 127.0.0.1 -port {cfg.port}"
                )

            game.add_game_args(
                f"+name AGENT{cfg.rank} +colorset {cfg.rank % 8}"
            )
        else:
            game.set_mode(vzd.Mode.PLAYER)

        return game

    # -- observation ----------------------------------------------------------

    def _get_screen(self) -> Optional[np.ndarray]:
        state = self.game.get_state()
        return state.screen_buffer if state is not None else None

    def _preprocess(self, frame: Optional[np.ndarray]) -> np.ndarray:
        """RGB screen → (3, H, W) uint8 at obs_resolution."""
        h, w = self.cfg.obs_resolution

        if frame is None:
            return np.zeros((3, h, w), dtype=np.uint8)

        frame = np.asarray(frame)

        if frame.ndim == 3 and frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))

        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        return np.transpose(frame, (2, 0, 1)).astype(np.uint8)

    def _stacked_obs(self) -> np.ndarray:
        """Returns (frame_stack * 3, H, W) uint8 array."""
        return np.concatenate(list(self.frames), axis=0)

    # -- reward ---------------------------------------------------------------

    def _get_var(self, var: vzd.GameVariable) -> float:
        try:
            return float(self.game.get_game_variable(var))
        except Exception:
            return 0.0

    def _snapshot_vars(self) -> dict:
        return {v: self._get_var(v) for v in GAME_VARS}

    def _compute_reward(self, cur: dict, dead: bool, action_idx: int) -> float:
        prev = self._prev_vars
        rc = self.cfg.reward

        d_frag = (cur[vzd.GameVariable.FRAGCOUNT] - prev[vzd.GameVariable.FRAGCOUNT]) + \
                 (cur[vzd.GameVariable.KILLCOUNT] - prev[vzd.GameVariable.KILLCOUNT])
        d_damage = cur[vzd.GameVariable.DAMAGECOUNT]           - prev[vzd.GameVariable.DAMAGECOUNT]
        d_taken  = cur[vzd.GameVariable.DAMAGE_TAKEN]          - prev[vzd.GameVariable.DAMAGE_TAKEN]
        d_hit    = cur[vzd.GameVariable.HITCOUNT]              - prev[vzd.GameVariable.HITCOUNT]
        d_item   = cur[vzd.GameVariable.ITEMCOUNT]             - prev[vzd.GameVariable.ITEMCOUNT]
        d_health = cur[vzd.GameVariable.HEALTH]                - prev[vzd.GameVariable.HEALTH]
        d_ammo   = cur[vzd.GameVariable.SELECTED_WEAPON_AMMO] - prev[vzd.GameVariable.SELECTED_WEAPON_AMMO]

        reward = rc.living
        reward += d_frag * rc.frag
        reward += max(0.0, d_damage) * rc.damage
        reward += max(0.0, d_taken)  * rc.damage_taken
        reward += max(0.0, d_hit)    * rc.hit
        reward += max(0.0, d_item)   * rc.item
        reward += max(0.0, d_ammo)   * rc.ammo
        if d_health > 0:
            reward += d_health * rc.health
        if dead:
            reward -= rc.death

        # penalty for shooting and missing
        is_attack = self.actions[action_idx][0] == 1  # ATTACK is the first button
        if is_attack and d_hit == 0:
            reward -= rc.miss

        return float(np.clip(reward, rc.clip_min, rc.clip_max))

    # -- metrics --------------------------------------------------------------

    def _build_info(self, cur: dict, action_idx: int) -> dict:
        """Build info dict with raw and derived metrics."""
        frags = cur[vzd.GameVariable.FRAGCOUNT]
        deaths = cur[vzd.GameVariable.DEATHCOUNT]
        damage_dealt = cur[vzd.GameVariable.DAMAGECOUNT]
        damage_taken = cur[vzd.GameVariable.DAMAGE_TAKEN]
        hits = cur[vzd.GameVariable.HITCOUNT]
        items = cur[vzd.GameVariable.ITEMCOUNT]

        # track attack actions for accuracy approximation
        is_attack = self.actions[action_idx][0] == 1
        if is_attack:
            self._attack_steps += 1

        return {
            # raw game stats
            "frags":        frags,
            "deaths":       deaths,
            "damage_dealt": damage_dealt,
            "damage_taken": damage_taken,
            "hits":         hits,
            "items":        items,
            "health":       cur[vzd.GameVariable.HEALTH],
            "armor":        cur[vzd.GameVariable.ARMOR],
            "ammo":         cur[vzd.GameVariable.SELECTED_WEAPON_AMMO],

            # derived performance metrics
            "kd_ratio":       frags / max(deaths, 1),
            "damage_ratio":   damage_dealt / max(damage_taken, 1),
            "hit_rate":       hits / max(self._attack_steps, 1),

            # episode tracking
            "episode_steps":      self._episode_steps,
            "survival_steps":     self._steps_since_death,
            "attack_steps":       self._attack_steps,
        }

    # -- recording ------------------------------------------------------------

    def _grab_frame(self):
        if not self.cfg.record:
            return
        if len(self.recorded_frames) >= self.cfg.max_rec_frames:
            return
        buf = self._get_screen()
        if buf is None:
            return
        buf = np.asarray(buf)
        if buf.ndim == 3 and buf.shape[0] == 3:
            buf = np.transpose(buf, (1, 2, 0))
        if buf.ndim == 3 and buf.shape[-1] >= 3:
            self.recorded_frames.append(
                np.ascontiguousarray(buf[..., :3], dtype=np.uint8).copy()
            )

    def start_recording(self):
        self.recorded_frames = []
        self.cfg.record = True

    def stop_recording(self) -> list:
        self.cfg.record = False
        return self.recorded_frames

    # -- core interface -------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Start a new episode. Returns stacked RGB observation (uint8)."""
        self.game.new_episode()
        self._prev_vars = self._snapshot_vars()
        self._episode_steps = 0
        self._steps_since_death = 0
        self._attack_steps = 0

        frame = self._preprocess(self._get_screen())
        self.frames.clear()
        for _ in range(self.cfg.frame_stack):
            self.frames.append(frame)

        self._grab_frame()
        return self._stacked_obs()

    def step(self, action_idx: int):
        """
        Execute action, return (obs, reward, done, info).

        On death the player respawns automatically (episode continues).
        Episode ends only on timeout.
        """
        self.game.make_action(self.actions[action_idx], self.cfg.frame_skip)
        done = self.game.is_episode_finished()
        dead = self.game.is_player_dead()

        cur = self._snapshot_vars()
        reward = self._compute_reward(cur, dead, action_idx)  # passa action_idx
        self._prev_vars = cur

        # update step counters
        self._episode_steps += 1
        self._steps_since_death += 1

        # respawn on death (episode continues until timeout)
        if dead and not done:
            self.game.respawn_player()
            self._prev_vars = self._snapshot_vars()
            self._steps_since_death = 0

        if done:
            frame = np.zeros((3, *self.cfg.obs_resolution), dtype=np.uint8)
        else:
            frame = self._preprocess(self._get_screen())
            self._grab_frame()

        self.frames.append(frame)

        info = self._build_info(cur, action_idx)

        return self._stacked_obs(), reward, done, info

    def initial_obs(self) -> np.ndarray:
        """
        Get initial observation after game.init() without calling new_episode().
        In networked mode, init() already starts the first episode.
        """
        self._prev_vars = self._snapshot_vars()
        self._episode_steps = 0
        self._steps_since_death = 0
        self._attack_steps = 0

        frame = self._preprocess(self._get_screen())
        self.frames.clear()
        for _ in range(self.cfg.frame_stack):
            self.frames.append(frame)
        self._grab_frame()
        return self._stacked_obs()

    @property
    def obs_shape(self) -> tuple:
        """Shape of a single observation: (in_channels, H, W)."""
        h, w = self.cfg.obs_resolution
        return (self.in_channels, h, w)

    def close(self):
        try:
            self.game.close()
        except Exception:
            pass