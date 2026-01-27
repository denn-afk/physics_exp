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
    ddq = dyn.ddq(q, dq, Q)          # = solve(M, Q - h)
    dq = dq + dt * ddq               # dq_free

    # contact impulse (PGS)
    dq, contact = resolve_toe_heel_pgs(dyn, q, dq, dt, mu=mu, beta=beta, iters=iters)
    # integrate
    q = q + dt * dq
    q = wrap_angles(q)

    return q, dq, contact