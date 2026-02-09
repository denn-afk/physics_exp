import time
from contact_pgs import resolve_toe_heel_pgs
import numpy as np

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

def wrap_angles(q):
    q_wrapped = q.copy()
    for idx in range(2, len(q)):
        q_wrapped[idx] = wrap_angle(q[idx])
    return q_wrapped

def physics_step(dyn, q, dq, dt, Q=None, mu=0.8, beta=0.2, iters=20):
    """semi-implicit Euler + impulse contact correction"""
    q = np.asarray(q, float); dq = np.asarray(dq, float)
    if Q is None: Q = np.zeros_like(dq)

    # free dynamics
    # idx_act = np.arange(3, len(q))  # actuated joint indices (exclude floating base)
    # free dynamics
    Qeff = Q + joint_friction(dq, b=0.1, tau_c=10.0, v0=2e-2, base_dofs=3)

    ddq = dyn.ddq(q, dq, Qeff)          # = solve(M, Q - h)
    dq = dq + dt * ddq               # dq_free

    # contact impulse (PGS)
    dq, contact = resolve_toe_heel_pgs(dyn, q, dq, dt, mu=mu, beta=beta, iters=iters)
    # integrate
    q = q + dt * dq
    q = wrap_angles(q)

    return q, dq, contact

def joint_friction(dq, b=0.0, tau_c=0.0, v0=1e-2, base_dofs=3):
    """
    dq: (n,) generalized velocities
    b: viscous coeff (scalar)
    tau_c: coulomb magnitude (scalar)
    v0: smoothing speed for tanh(sign)
    base_dofs: first k DOFs are floating base (no joint friction)
    """
    Qf = np.zeros_like(dq)
    if dq.size <= base_dofs:
        return Qf
    v = dq[base_dofs:]
    Qf[base_dofs:] = -b * v - tau_c * np.tanh(v / v0)
    return Qf
