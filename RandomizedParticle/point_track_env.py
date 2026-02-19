import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError("Please install gymnasium: pip install gymnasium") from e


@dataclass
class DomainRandomizationCfg:
    mass_range: Tuple[float, float] = (0.5, 7.0)     # m
    damping_range: Tuple[float, float] = (0.0, 3.0)  # c (linear damping)
    # If you later want more params: wind/bias, dt jitter, etc.


class PointParticleTrackEnv(gym.Env):
    """
    2D point particle with inertia.
    Task: track a desired trajectory p*(t), v*(t).

    State: s = [px, py, vx, vy]
    Action: a = [ax_cmd, ay_cmd] (bounded), interpreted as force-like command.
            True acceleration: u = a/m - c*v
    Observation: [p, v, p*, v*, t_norm]
    Reward: - (w_p*||p-p*||^2 + w_v*||v-v*||^2 + w_u*||a||^2)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.02,
        horizon: int = 300,
        a_max: float = 10.0,
        dr_cfg: DomainRandomizationCfg = DomainRandomizationCfg(),
        traj: str = "circle",
        # reward weights
        w_pos: float = 5.0,
        w_vel: float = 0.5,
        w_act: float = 0.02,
        # optional termination bounds
        pos_bound: float = 5.0,
        vel_bound: float = 20.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.a_max = float(a_max)
        self.dr_cfg = dr_cfg
        self.traj = traj

        self.w_pos = float(w_pos)
        self.w_vel = float(w_vel)
        self.w_act = float(w_act)

        self.pos_bound = float(pos_bound)
        self.vel_bound = float(vel_bound)

        self.np_random = np.random.default_rng(seed)

        # Action space: bounded accel/force command
        self.action_space = spaces.Box(
            low=-self.a_max, high=self.a_max, shape=(2,), dtype=np.float32
        )

        # Observation space: [p(2), v(2), p*(2), v*(2), t_norm(1)] -> 9 dims
        obs_high = np.array([np.inf] * 9, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Internal
        self.t = 0
        self.state = np.zeros(4, dtype=np.float64)  # [px,py,vx,vy]
        self.mass = 1.0
        self.damping = 0.0

        # Trajectory params (circle)
        self._R = 1.0
        self._omega = 2.0 * math.pi / (self.horizon * self.dt)  # one loop per episode
        self._center = np.array([0.0, 0.0], dtype=np.float64)

        self._fig = None
        self._ax = None
        self._particle = None
        self._target = None
        self._traj_line = None
        self._target_line = None
        self._history = []
        self._target_history = []

    # -------- desired trajectory --------
    def desired(self, t_step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (p*, v*) at discrete time step t_step."""
        t = t_step * self.dt

        if self.traj == "circle":
            # p*(t) = c + R [cos(wt), sin(wt)]
            ct = math.cos(self._omega * t)
            st = math.sin(self._omega * t)
            p_star = self._center + self._R * np.array([ct, st], dtype=np.float64)
            v_star = self._R * self._omega * np.array([-st, ct], dtype=np.float64)
            return p_star, v_star

        elif self.traj == "lissajous":
            # A simple Lissajous curve
            A, B = 1.0, 1.0
            wx, wy = 1.5 * self._omega, 2.0 * self._omega
            px = A * math.sin(wx * t + 0.3)
            py = B * math.sin(wy * t)
            vx = A * wx * math.cos(wx * t + 0.3)
            vy = B * wy * math.cos(wy * t)
            return np.array([px, py], float), np.array([vx, vy], float)

        else:
            raise ValueError(f"Unknown traj='{self.traj}'")

    # -------- gym API --------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.t = 0

        # Domain randomization
        m_lo, m_hi = self.dr_cfg.mass_range
        c_lo, c_hi = self.dr_cfg.damping_range
        self.mass = float(self.np_random.uniform(m_lo, m_hi))
        self.damping = float(self.np_random.uniform(c_lo, c_hi))

        # Initialize near desired start with small noise
        p0_star, v0_star = self.desired(0)
        p0 = p0_star + self.np_random.normal(scale=0.05, size=(2,))
        v0 = v0_star + self.np_random.normal(scale=0.05, size=(2,))
        self.state = np.array([p0[0], p0[1], v0[0], v0[1]], dtype=np.float64)

        obs = self._get_obs()
        info = {
            "z_true": np.array([self.mass, self.damping], dtype=np.float32),
            "mass": self.mass,
            "damping": self.damping,
        }
        self._history = []
        self._target_history = []

        return obs, info
    
    def render(self):
        px, py, vx, vy = self.state
        p = np.array([px, py])

        p_star, _ = self.desired(self.t)

        self._history.append(p.copy())
        self._target_history.append(p_star.copy())

        if self._fig is None:
            self._fig, self._ax = plt.subplots()
            self._ax.set_aspect("equal")
            self._ax.set_xlim(-2, 2)
            self._ax.set_ylim(-2, 2)

            self._particle, = self._ax.plot([], [], "bo", markersize=8, label="particle")
            self._target, = self._ax.plot([], [], "ro", markersize=8, label="target")

            self._traj_line, = self._ax.plot([], [], "b-", alpha=0.3)
            self._target_line, = self._ax.plot([], [], "r--", alpha=0.3)

            self._ax.legend()
            plt.ion()
            plt.show()

        hist = np.array(self._history)
        target_hist = np.array(self._target_history)

        self._particle.set_data([p[0]], [p[1]])
        self._target.set_data([p_star[0]], [p_star[1]])

        self._traj_line.set_data([hist[:, 0]], [hist[:, 1]])
        self._target_line.set_data([target_hist[:, 0]], [target_hist[:, 1]])

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -self.a_max, self.a_max)

        px, py, vx, vy = self.state
        v = np.array([vx, vy], dtype=np.float64)

        # True acceleration
        u = (action / self.mass) - self.damping * v

        # Semi-implicit Euler
        v_next = v + self.dt * u
        p_next = np.array([px, py], dtype=np.float64) + self.dt * v_next

        self.state = np.array([p_next[0], p_next[1], v_next[0], v_next[1]], dtype=np.float64)
        self.t += 1

        # Desired
        p_star, v_star = self.desired(self.t)

        # Reward
        p_err = p_next - p_star
        v_err = v_next - v_star
        reward = -(
            self.w_pos * float(np.dot(p_err, p_err))
            + self.w_vel * float(np.dot(v_err, v_err))
            + self.w_act * float(np.dot(action, action))
        )

        # Termination
        terminated = False
        if (np.linalg.norm(p_next) > self.pos_bound) or (np.linalg.norm(v_next) > self.vel_bound):
            terminated = True

        truncated = (self.t >= self.horizon)

        obs = self._get_obs()

        info: Dict[str, Any] = {
            "z_true": np.array([self.mass, self.damping], dtype=np.float32),
            "p_star": p_star.astype(np.float32),
            "v_star": v_star.astype(np.float32),
            "p_err": p_err.astype(np.float32),
            "v_err": v_err.astype(np.float32),
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        px, py, vx, vy = self.state
        p_star, v_star = self.desired(self.t)
        t_norm = self.t / max(1, self.horizon)

        obs = np.array(
            [px, py, vx, vy, p_star[0], p_star[1], v_star[0], v_star[1], t_norm],
            dtype=np.float32,
        )
        return obs

if __name__ == "__main__":
    env = PointParticleTrackEnv(traj="circle", seed=0)
    for _ in range(10000):  # run 5 episodes
        obs, info = env.reset()

        kp, kd = 20.0, 8.0
        for _ in range(env.horizon):
            px, py, vx, vy, ptx, pty, vtx, vty, _ = obs

            p = np.array([px, py])
            v = np.array([vx, vy])
            p_star = np.array([ptx, pty])
            v_star = np.array([vtx, vty])

            a = kp * (p_star - p) + kd * (v_star - v)

            obs, r, terminated, truncated, info = env.step(a)
            env.render()

            if terminated or truncated:
                break

    env.close()

