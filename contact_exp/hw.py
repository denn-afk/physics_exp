import numpy as np

class TwoLinkPlanar:
    """
    Planar 2-link (2 DOF), base fixed at origin.
    y-up world.
    """

    def __init__(self, l1=0.5, l2=0.5, m1=1.0, m2=1.0, I1=None, I2=None, g=9.81):
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.g = float(g)

        # COM distances from proximal joint (midpoint by default)
        self.lc1 = 0.5 * self.l1
        self.lc2 = 0.5 * self.l2

        # Inertia about COM (thin rod about center if not provided)
        if I1 is None:
            I1 = (self.m1 * self.l1**2) / 12.0
        if I2 is None:
            I2 = (self.m2 * self.l2**2) / 12.0
        self.I1 = float(I1)
        self.I2 = float(I2)

    # ---------- kinematics ----------
    def fk(self, q):
        q1, q2 = q
        x = self.l1*np.cos(q1) + self.l2*np.cos(q1+q2)
        y = self.l1*np.sin(q1) + self.l2*np.sin(q1+q2)
        return np.array([x, y])

    def jacobian(self, q):
        q1, q2 = q
        s1 = np.sin(q1); c1 = np.cos(q1)
        s12 = np.sin(q1+q2); c12 = np.cos(q1+q2)

        # foot position:
        # x = l1 c1 + l2 c12
        # y = l1 s1 + l2 s12
        dx_dq1 = -self.l1*s1 - self.l2*s12
        dx_dq2 = -self.l2*s12
        dy_dq1 =  self.l1*c1 + self.l2*c12
        dy_dq2 =  self.l2*c12

        return np.array([[dx_dq1, dx_dq2],
                         [dy_dq1, dy_dq2]])

    # ---------- dynamics: M, C, G ----------
    def M(self, q):
        q1, q2 = q
        c2 = np.cos(q2)

        m1, m2 = self.m1, self.m2
        l1, lc1, lc2 = self.l1, self.lc1, self.lc2
        I1, I2 = self.I1, self.I2

        # standard 2-link inertia matrix
        M11 = I1 + I2 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2)
        M12 = I2 + m2*(lc2**2 + l1*lc2*c2)
        M22 = I2 + m2*lc2**2

        return np.array([[M11, M12],
                         [M12, M22]])

    def C(self, q, dq):
        """
        Coriolis/centrifugal vector (2,)
        Using common form:
        C1 = -m2*l1*lc2*sin(q2) * (2*dq1*dq2 + dq2^2)
        C2 =  m2*l1*lc2*sin(q2) * (dq1^2)
        """
        q1, q2 = q
        dq1, dq2 = dq
        s2 = np.sin(q2)

        m2 = self.m2
        l1, lc2 = self.l1, self.lc2

        h = m2*l1*lc2*s2
        C1 = -h*(2*dq1*dq2 + dq2*dq2)
        C2 =  h*(dq1*dq1)
        return np.array([C1, C2])

    def G(self, q):
        """
        Gravity torque vector (2,)
        y-up => potential is +m*g*y
        """
        q1, q2 = q
        m1, m2 = self.m1, self.m2
        g = self.g
        l1, lc1, lc2 = self.l1, self.lc1, self.lc2

        # COM heights:
        # y1 = lc1 sin q1
        # y2 = l1 sin q1 + lc2 sin(q1+q2)
        # G = dV/dq
        G1 = (m1*lc1 + m2*l1)*g*np.cos(q1) + m2*lc2*g*np.cos(q1+q2)
        G2 = m2*lc2*g*np.cos(q1+q2)
        return np.array([G1, G2])

    def forward_dynamics(self, q, dq, tau, Jt_lambda=None):
        """
        qdd = M^{-1} ( tau + J^T lambda - C - G )
        Jt_lambda: shape (2,) joint-space generalized force from contact, optional
        """
        M = self.M(q)
        rhs = tau - self.C(q, dq) - self.G(q)
        if Jt_lambda is not None:
            rhs = rhs + Jt_lambda
        qdd = np.linalg.solve(M, rhs)
        return qdd
