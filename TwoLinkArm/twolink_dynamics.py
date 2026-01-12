# twolink_dynamics.py
# Recommended version (NO shortcuts):
# - Explicit link lengths l1, l2
# - COM offsets c1, c2 (can be <= li)
# - Lambdified:
#   * M(q)
#   * h(q, dq)   where M(q) ddq + h(q,dq) = tau
#   * G(q)
#   * H_cc(q,dq)
#   * Link endpoint positions pL1(q), pEE(q)
#   * Link endpoint linear Jacobians J_L1(q), J_EE(q)  (2x2)

import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
import numpy as np


def build_twolink_numpy_class():
    # ===== time & generalized coords =====
    t = sp.symbols("t")
    q1, q2 = dynamicsymbols("q1 q2")
    q = sp.Matrix([q1, q2])
    dq = q.diff(t)
    ddq = dq.diff(t)
    dq1, dq2 = dq[0], dq[1]

    # ===== parameters =====
    # link lengths l1,l2 and COM offsets c1,c2
    l1, l2, c1, c2 = sp.symbols("l1 l2 c1 c2", real=True)
    m1, m2 = sp.symbols("m1 m2", real=True, positive=True)
    I1, I2 = sp.symbols("I1 I2", real=True, positive=True)
    g = sp.symbols("g", real=True, positive=True)

    # ===== COM positions =====
    x1 = c1 * sp.cos(q1)
    y1 = c1 * sp.sin(q1)

    x2 = l1 * sp.cos(q1) + c2 * sp.cos(q1 + q2)
    y2 = l1 * sp.sin(q1) + c2 * sp.sin(q1 + q2)

    p1 = sp.Matrix([x1, y1])
    p2 = sp.Matrix([x2, y2])

    # ===== Link endpoint positions (NOT COM) =====
    # link1 end (joint2 position)
    xL1 = l1 * sp.cos(q1)
    yL1 = l1 * sp.sin(q1)
    pL1 = sp.Matrix([xL1, yL1])

    # link2 end (end-effector)
    xEE = l1 * sp.cos(q1) + l2 * sp.cos(q1 + q2)
    yEE = l1 * sp.sin(q1) + l2 * sp.sin(q1 + q2)
    pEE = sp.Matrix([xEE, yEE])

    # ===== Jacobians & velocities =====
    J1 = p1.jacobian(q)     # COM1 linear Jacobian (2x2)
    J2 = p2.jacobian(q)     # COM2 linear Jacobian (2x2)
    v1 = J1 * dq
    v2 = J2 * dq

    # Endpoint linear Jacobians
    J_L1 = pL1.jacobian(q)  # 2x2
    J_EE = pEE.jacobian(q)  # 2x2

    # ===== kinetic energy =====
    T1 = sp.simplify(sp.Rational(1, 2) * m1 * (v1.dot(v1)) + sp.Rational(1, 2) * I1 * dq1**2)
    T2 = sp.simplify(
        sp.Rational(1, 2) * m2 * (v2.dot(v2)) + sp.Rational(1, 2) * I2 * (dq1 + dq2) ** 2
    )
    T = sp.simplify(T1 + T2)

    # ===== potential energy =====
    V = sp.simplify(m1 * g * y1 + m2 * g * y2)

    L = sp.simplify(T - V)

    # ===== Lagrange equations (no external forces): d/dt(dL/ddq) - dL/dq = 0 =====
    dL_dq = sp.Matrix([L]).jacobian(q).T
    dL_ddq = sp.Matrix([L]).jacobian(dq).T
    eq = sp.simplify(dL_ddq.diff(t) - dL_dq)  # = 0

    # ===== Extract M and h: eq = M*ddq + h =====
    M = sp.simplify(eq.jacobian(ddq))
    h = sp.simplify(eq - M * ddq)

    G = sp.simplify(h.subs({dq1: 0, dq2: 0}))
    H_cc = sp.simplify(h - G)

    sp.pprint(M)
    sp.pprint(h)
    sp.pprint(G)
    sp.pprint(H_cc)

    # ---- plain symbols for lambdify (replace dynamicsymbols) ----
    q1s, q2s, dq1s, dq2s = sp.symbols("q1 q2 dq1 dq2", real=True)
    subs_map = {q1: q1s, q2: q2s, dq1: dq1s, dq2: dq2s}

    M_s = sp.simplify(M.subs(subs_map))
    h_s = sp.simplify(h.subs(subs_map))
    G_s = sp.simplify(G.subs(subs_map))
    Hcc_s = sp.simplify(H_cc.subs(subs_map))

    pL1_s = sp.simplify(pL1.subs(subs_map))
    pEE_s = sp.simplify(pEE.subs(subs_map))
    JL1_s = sp.simplify(J_L1.subs(subs_map))
    JEE_s = sp.simplify(J_EE.subs(subs_map))

    # argument orders
    args_M = (q1s, q2s, l1, l2, c1, c2, m1, m2, I1, I2, g)
    args_h = (q1s, q2s, dq1s, dq2s, l1, l2, c1, c2, m1, m2, I1, I2, g)
    args_J = (q1s, q2s, l1, l2)  # endpoints/J only depend on geometry

    # lambdified functions
    _M_fun = sp.lambdify(args_M, M_s, modules="numpy")
    _h_fun = sp.lambdify(args_h, h_s, modules="numpy")
    _G_fun = sp.lambdify(args_M, G_s, modules="numpy")
    _Hcc_fun = sp.lambdify(args_h, Hcc_s, modules="numpy")

    _pL1_fun = sp.lambdify(args_J, pL1_s, modules="numpy")
    _pEE_fun = sp.lambdify(args_J, pEE_s, modules="numpy")
    _JL1_fun = sp.lambdify(args_J, JL1_s, modules="numpy")
    _JEE_fun = sp.lambdify(args_J, JEE_s, modules="numpy")

    class TwoLinkDynamics:
        """
        Numeric (numpy) dynamics + endpoint kinematics.
        All methods accept q,dq,tau as array-like shape (2,).
        """

        def __init__(self, l1_, l2_, c1_, c2_, m1_, m2_, I1_, I2_, g_=9.81, dtype=float):
            self.l1 = dtype(l1_)
            self.l2 = dtype(l2_)
            self.c1 = dtype(c1_)
            self.c2 = dtype(c2_)
            self.m1 = dtype(m1_)
            self.m2 = dtype(m2_)
            self.I1 = dtype(I1_)
            self.I2 = dtype(I2_)
            self.g = dtype(g_)
            self.dtype = dtype

        def _param_tuple_M(self):
            return (self.l1, self.l2, self.c1, self.c2, self.m1, self.m2, self.I1, self.I2, self.g)

        def _param_tuple_J(self):
            return (self.l1, self.l2)

        # ----- dynamics -----
        def M(self, q):
            q1v, q2v = map(self.dtype, q)
            out = np.array(_M_fun(q1v, q2v, *self._param_tuple_M()), dtype=self.dtype)
            return out.reshape(2, 2)

        def h(self, q, dq):
            q1v, q2v = map(self.dtype, q)
            dq1v, dq2v = map(self.dtype, dq)
            out = np.array(_h_fun(q1v, q2v, dq1v, dq2v, *self._param_tuple_M()), dtype=self.dtype)
            return out.reshape(2,)

        def G(self, q):
            q1v, q2v = map(self.dtype, q)
            out = np.array(_G_fun(q1v, q2v, *self._param_tuple_M()), dtype=self.dtype)
            return out.reshape(2,)

        def H_cc(self, q, dq):
            q1v, q2v = map(self.dtype, q)
            dq1v, dq2v = map(self.dtype, dq)
            out = np.array(_Hcc_fun(q1v, q2v, dq1v, dq2v, *self._param_tuple_M()), dtype=self.dtype)
            return out.reshape(2,)

        def ddq(self, q, dq, tau):
            """
            Solve: M(q) ddq + h(q,dq) = tau  -> ddq = solve(M, tau - h)
            """
            Mmat = self.M(q)
            hvec = self.h(q, dq)
            tau = np.asarray(tau, dtype=self.dtype).reshape(2,)
            return np.linalg.solve(Mmat, tau - hvec)

        # ----- endpoint kinematics -----
        def link1_end(self, q):
            q1v, q2v = map(self.dtype, q)
            out = np.array(_pL1_fun(q1v, q2v, *self._param_tuple_J()), dtype=self.dtype)
            return out.reshape(2,)

        def ee(self, q):
            q1v, q2v = map(self.dtype, q)
            out = np.array(_pEE_fun(q1v, q2v, *self._param_tuple_J()), dtype=self.dtype)
            return out.reshape(2,)

        def J_link1_end(self, q):
            q1v, q2v = map(self.dtype, q)
            out = np.array(_JL1_fun(q1v, q2v, *self._param_tuple_J()), dtype=self.dtype)
            return out.reshape(2, 2)

        def J_ee(self, q):
            q1v, q2v = map(self.dtype, q)
            out = np.array(_JEE_fun(q1v, q2v, *self._param_tuple_J()), dtype=self.dtype)
            return out.reshape(2, 2)

    return TwoLinkDynamics


# Instantiate at import time
TwoLinkDynamics = build_twolink_numpy_class()
