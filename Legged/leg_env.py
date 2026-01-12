# leg_env.py
import numpy as np
from legged_dynamics import LegDynamicsCached
from contact_pgs import resolve_toe_heel_pgs

class LegEnv:
    def __init__(self, dyn: LegDynamicsCached,
                 dt=0.01, mu=0.8, beta=0.2, iters=25,
                 tau_max=80.0):
        self.dyn = dyn
        self.dt = dt
        self.mu = mu
        self.beta = beta
        self.iters = iters
        self.tau_max = tau_max

        self.q = None
        self.dq = None
        self.t = 0.0

    def reset(self, randomize=False):
        # 一个比较合理的初始站立姿态
        if not randomize:
            q = np.array([0.0, 2.0, 0.0, 0.3, -0.6, 0.2], dtype=float)
            dq = np.zeros(6, dtype=float)
        else:
            # 可以加一点 random 初始化
            q = np.array([0.0, 2.0, 0.0, 0.3, -0.6, 0.2], dtype=float)
            q += np.random.randn(6) * np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
            dq = np.random.randn(6) * 0.5

        self.q = q
        self.dq = dq
        self.t = 0.0
        return self._get_obs()

    def _get_obs(self):
        # 观测可以直接用 [q, dq]
        return np.concatenate([self.q, self.dq], axis=0)

    def step(self, tau):
        """单步: semi-implicit Euler + PGS."""
        q = self.q
        dq = self.dq
        dt = self.dt

        tau = np.asarray(tau, float).reshape(-1)
        tau = np.clip(tau, -self.tau_max, self.tau_max)

        # free dynamics
        ddq_free = self.dyn.ddq(q, dq, tau)       # solve M ddq = tau - h
        dq_free = dq + dt * ddq_free

        # contact impulse correction
        dq_plus, contact = resolve_toe_heel_pgs(
            self.dyn, q, dq_free, dt,
            mu=self.mu, beta=self.beta, iters=self.iters
        )

        q_next = q + dt * dq_plus

        # 更新 env 内部状态
        self.q = q_next
        self.dq = dq_plus
        self.t += dt

        obs_next = self._get_obs()

        # 这里 reward / done 你可以随便定义，先只用 done 做安全 reset
        done = False
        info = {"contact": contact}

        # 简单 safety：摔倒 / 爆 NaN 就结束
        if (not np.isfinite(q_next).all()
            or not np.isfinite(dq_plus).all()
            or q_next[1] < 0.0
            or np.abs(q_next).max() > 10.0
            or np.abs(dq_plus).max() > 100.0):
            done = True

        return obs_next, done, info
