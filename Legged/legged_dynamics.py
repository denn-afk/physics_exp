import os, pickle
import sympy as sym
from sympy import Matrix, sin, cos
import numpy as np

import numpy as np

def _clip_segment_to_rect(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    """
    Clip segment to axis-aligned rectangle [xmin,xmax]x[ymin,ymax].
    Returns (cx0,cy0,cx1,cy1, ok).
    Liang–Barsky algorithm.
    """
    dx = x1 - x0
    dy = y1 - y0

    # p_i and q_i for each boundary
    p0 = -dx; q0 = x0 - xmin
    p1 =  dx; q1 = xmax - x0
    p2 = -dy; q2 = y0 - ymin
    p3 =  dy; q3 = ymax - y0

    u0, u1 = 0.0, 1.0

    # Unrolled loop is a tiny bit faster in Python
    for pi, qi in ((p0, q0), (p1, q1), (p2, q2), (p3, q3)):
        if abs(pi) < 1e-12:
            if qi < 0:
                return x0, y0, x1, y1, False
        else:
            t = qi / pi
            if pi < 0:
                if t > u0: u0 = t
            else:
                if t < u1: u1 = t
            if u0 > u1:
                return x0, y0, x1, y1, False

    cx0 = x0 + u0 * dx
    cy0 = y0 + u0 * dy
    cx1 = x0 + u1 * dx
    cy1 = y0 + u1 * dy
    return cx0, cy0, cx1, cy1, True


def _draw_line(img, x0, y0, x1, y1, color, thickness=1):
    """Fast line rasterization with clipping, vectorized stamping for thickness."""
    H, W, _ = img.shape

    # Reject NaN/inf early
    if not np.isfinite([x0, y0, x1, y1]).all():
        return

    # Clip segment to image bounds first
    x0, y0, x1, y1, ok = _clip_segment_to_rect(
        float(x0), float(y0), float(x1), float(y1),
        xmin=0.0, ymin=0.0, xmax=float(W - 1), ymax=float(H - 1)
    )
    if not ok:
        return

    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy))) + 1
    if steps <= 1:
        xi = int(round(x0))
        yi = int(round(y0))
        if 0 <= xi < W and 0 <= yi < H:
            img[yi, xi] = color
        return

    # Vectorized DDA (no Python per-point loop)
    xs = np.rint(np.linspace(x0, x1, steps)).astype(np.int32)
    ys = np.rint(np.linspace(y0, y1, steps)).astype(np.int32)

    # Optional: remove consecutive duplicates to reduce work (common for shallow slopes)
    if xs.size >= 2:
        m = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
        # keep first
        idx = np.empty(xs.size, dtype=bool)
        idx[0] = True
        idx[1:] = m
        xs = xs[idx]
        ys = ys[idx]

    # Thickness stamping: loop over K^2 offsets (small), vectorized over all points
    r = int(thickness // 2)
    if r <= 0:
        m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        if np.any(m):
            img[ys[m], xs[m]] = color
        return

    off = np.arange(-r, r + 1, dtype=np.int32)

    # For each offset pair, paint all shifted points at once
    for oy in off:
        y = ys + oy
        my = (y >= 0) & (y < H)
        if not np.any(my):
            continue
        for ox in off:
            x = xs + ox
            m = my & (x >= 0) & (x < W)
            if np.any(m):
                img[y[m], x[m]] = color


def _fill_convex_poly(img, pts, color):
    """
    Very fast fill for convex polygon using half-space test on bbox grid.
    pts: (N,2) pixel coords (x right, y down).
    """
    H, W, _ = img.shape
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] < 3 or (not np.isfinite(pts).all()):
        return

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    if xmax < 0 or xmin > (W - 1) or ymax < 0 or ymin > (H - 1):
        return

    x0 = int(max(0, np.floor(xmin)))
    x1 = int(min(W - 1, np.ceil(xmax)))
    y0 = int(max(0, np.floor(ymin)))
    y1 = int(min(H - 1, np.ceil(ymax)))
    if x1 < x0 or y1 < y0:
        return

    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)  # (Hb, Wb)

    A = pts
    B = np.roll(pts, -1, axis=0)
    E = B - A

    # cross(E, P-A) = ex*(Y-ay) - ey*(X-ax)
    ax = A[:, 0][:, None, None]
    ay = A[:, 1][:, None, None]
    ex = E[:, 0][:, None, None]
    ey = E[:, 1][:, None, None]

    # Broadcast over edges
    cross = ex * (Y[None, :, :] - ay) - ey * (X[None, :, :] - ax)

    eps = 1e-6  # float32-friendly
    inside = np.all(cross >= -eps, axis=0) | np.all(cross <= eps, axis=0)

    block = img[y0:y1+1, x0:x1+1]
    block[inside] = color


def _world_to_pixel(Pw, W, H, xlim, ylim):
    """Pw: (...,2) in world; returns (...,2) in pixel float (x right, y down)."""
    Pw = np.asarray(Pw, dtype=np.float64)
    xmin, xmax = xlim
    ymin, ymax = ylim

    sx = (W - 1) / (xmax - xmin + 1e-12)
    sy = (H - 1) / (ymax - ymin + 1e-12)

    px = (Pw[..., 0] - xmin) * sx
    # map y: [ymin,ymax] -> [H-1,0]
    py = (H - 1) - (Pw[..., 1] - ymin) * sy
    return np.stack([px, py], axis=-1)



def _Jdot_from_Jq(J, q, dq):
    """Compute Jdot = dJ/dt = Σ_k ∂J/∂q_k * dq_k (no ddq needed)."""
    Jdot = sym.zeros(J.rows, J.cols)
    for i in range(J.rows):
        for j in range(J.cols):
            # scalar -> 1x6 jacobian
            Jdot[i, j] = Matrix([J[i, j]]).jacobian(q) * dq
    return sym.simplify(Jdot)


def _build_symbolic_exprs():
    # ---- symbols (NOTE: keep gravity constant symbol as g0; don't overwrite) ----
    (m_body, m1, m2, m3,
     I_body, I1, I2, I3,
     l0, l1, l2, l3,
     c1, c2, c3,
     g0) = sym.symbols('M m1 m2 m3 I I1 I2 I3 l0 l1 l2 l3 c1 c2 c3 g')

    (x, y, thb, th1, th2, th3,
     dx, dy, dthb, dth1, dth2, dth3,
     ddx, ddy, ddthb, ddth1, ddth2, ddth3) = sym.symbols(
        'x y thb th1 th2 th3 dx dy dthb dth1 dth2 dth3 ddx ddy ddthb ddth1 ddth2 ddth3'
    )

    q  = Matrix([x, y, thb, th1, th2, th3])
    dq = Matrix([dx, dy, dthb, dth1, dth2, dth3])
    ddq= Matrix([ddx, ddy, ddthb, ddth1, ddth2, ddth3])

    p = [m_body, m1, m2, m3, I_body, I1, I2, I3, l0, l1, l2, l3, c1, c2, c3, g0]
    zp_params = list(q) + list(dq) + list(p)  # numeric arg order

    # ---- kinematics ----
    r0 = Matrix([x, y])
    ehat1 = Matrix([sin(thb), -cos(thb)])
    ehat2 = Matrix([sin(thb+th1), -cos(thb+th1)])
    ehat3 = Matrix([sin(thb+th1+th2), -cos(thb+th1+th2)])
    ehat4 = Matrix([cos(thb+th1+th2+th3), sin(thb+th1+th2+th3)])
    ehat5 = -ehat4
    ghat  = Matrix([0, -1])

    rA  = r0 + l0*ehat1
    rB  = rA + l1*ehat2
    rC  = rB + l2*ehat3
    rD  = rC + sym.Rational(1,2)*l3*ehat4   # Heel
    rE  = rC + sym.Rational(1,2)*l3*ehat5   # Toe

    rcb = r0
    rc1 = rA + c1*ehat2
    rc2 = rB + c2*ehat3
    rc3 = rC

    # time-derivative helper for vectors that depend on q only
    def ddt_vec(expr_qonly):
        return expr_qonly.jacobian(q) @ dq

    vcb = ddt_vec(rcb)
    vc1 = ddt_vec(rc1)
    vc2 = ddt_vec(rc2)
    vc3 = ddt_vec(rc3)

    # ---- energies ----
    T1 = sym.Rational(1,2)*m_body*(vcb.dot(vcb)) + sym.Rational(1,2)*I_body*(dthb**2)
    T2 = sym.Rational(1,2)*m1*(vc1.dot(vc1)) + sym.Rational(1,2)*I1*((dthb + dth1)**2)
    T3 = sym.Rational(1,2)*m2*(vc2.dot(vc2)) + sym.Rational(1,2)*I2*((dthb + dth1 + dth2)**2)
    T4 = sym.Rational(1,2)*m3*(vc3.dot(vc3)) + sym.Rational(1,2)*I3*((dthb + dth1 + dth2 + dth3)**2)
    KE = sym.simplify(T1 + T2 + T3 + T4)

    P1 = (-rcb.dot(ghat))*g0*m_body
    P2 = (-rc1.dot(ghat))*g0*m1
    P3 = (-rc2.dot(ghat))*g0*m2
    P4 = (-rc3.dot(ghat))*g0*m3
    PE = sym.simplify(P1 + P2 + P3 + P4)

    L = Matrix([KE - PE])

    # ---- EOM: EOM = d/dt(dL/ddq) - dL/dq ----
    dL_dq  = L.jacobian(q).T
    dL_ddq = L.jacobian(dq).T

    # Here we need d/dt(dL/ddq). Since dL_ddq depends on (q,dq),
    # time derivative is: ∂/∂q * dq + ∂/∂dq * ddq
    def ddt_general(expr_q_dq):
        return expr_q_dq.jacobian(q)@dq + expr_q_dq.jacobian(dq)@ddq

    print("Building EOM...")
    EOM = sym.simplify(ddt_general(dL_ddq) - dL_dq)  # = Q (tau + J^T f + ...)

    print("Building A, h, grav, coriolis...")
    A = sym.simplify(EOM.jacobian(ddq))  # mass matrix

    zero_ddq = {ddx:0, ddy:0, ddthb:0, ddth1:0, ddth2:0, ddth3:0}
    h = sym.simplify(EOM.subs(zero_ddq))  # bias: coriolis+gravity

    grav = sym.simplify(Matrix([PE]).jacobian(q).T)
    coriolis = sym.simplify(h - grav)

    print("Building contact Jacobians + Jdot...")
    # ---- contact Jacobians + Jdot ----
    JToe  = rD.jacobian(q)   # 2x6
    JHeel = rE.jacobian(q)   # 2x6
    JToedot  = _Jdot_from_Jq(JToe, q, dq)   # 2x6
    JHeeldot = _Jdot_from_Jq(JHeel, q, dq)  # 2x6

    keypoints = Matrix([[r0],[rA],[rB],[rC],[rD],[rE]]).reshape(6,2)

    return {
        "zp_params": zp_params,
        "p_syms": p,
        "A": A, "h": h, "grav": grav, "coriolis": coriolis,
        "keypoints": keypoints,
        "JToe": JToe, "JHeel": JHeel,
        "JToedot": JToedot, "JHeeldot": JHeeldot,
        "pToe": rD, "pHeel": rE,
    }


def _load_or_build(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    exprs = _build_symbolic_exprs()
    with open(cache_path, "wb") as f:
        pickle.dump(exprs, f, protocol=pickle.HIGHEST_PROTOCOL)
    return exprs


class LegDynamicsCached:
    """
    Cached SymPy -> numpy dynamics for your 6-DoF planar model.
    Cache file defaults to ./leg3link_sympy_cache.pkl (current folder).
    """

    def __init__(self, params_dict, cache_path=os.path.join(os.path.dirname(__file__), "leg3link_sympy_cache.pkl"), dtype=float):
        self.dtype = dtype
        self.cache_path = cache_path

        exprs = _load_or_build(self.cache_path)

        self._zp_params = exprs["zp_params"]
        self._p_syms = exprs["p_syms"]

        # parameter values in the exact symbol order
        # params_dict keys should match symbol names: "M","m1","m2","m3","I","I1",...,"g"
        self._p_vals = [dtype(params_dict[s.name]) for s in self._p_syms]

        # lambdify
        self._A_fun        = sym.lambdify(self._zp_params, exprs["A"], modules="numpy")
        self._h_fun        = sym.lambdify(self._zp_params, exprs["h"], modules="numpy")
        self._grav_fun     = sym.lambdify(self._zp_params, exprs["grav"], modules="numpy")
        self._coriolis_fun = sym.lambdify(self._zp_params, exprs["coriolis"], modules="numpy")

        self._keypoints_fun = sym.lambdify(self._zp_params, exprs["keypoints"], modules="numpy")
        self._JToe_fun      = sym.lambdify(self._zp_params, exprs["JToe"], modules="numpy")
        self._JHeel_fun     = sym.lambdify(self._zp_params, exprs["JHeel"], modules="numpy")
        self._JToedot_fun   = sym.lambdify(self._zp_params, exprs["JToedot"], modules="numpy")
        self._JHeeldot_fun  = sym.lambdify(self._zp_params, exprs["JHeeldot"], modules="numpy")
        self._pToe_fun      = sym.lambdify(self._zp_params, exprs["pToe"], modules="numpy")
        self._pHeel_fun     = sym.lambdify(self._zp_params, exprs["pHeel"], modules="numpy")

    def _pack(self, q, dq):
        q = np.asarray(q, dtype=self.dtype).reshape(6,)
        dq = np.asarray(dq, dtype=self.dtype).reshape(6,)
        return tuple(q.tolist() + dq.tolist() + self._p_vals)

    # ---- dynamics ----
    def A(self, q, dq):
        return np.array(self._A_fun(*self._pack(q,dq)), dtype=self.dtype).reshape(6,6)

    def h(self, q, dq):
        return np.array(self._h_fun(*self._pack(q,dq)), dtype=self.dtype).reshape(6,)

    def grav(self, q, dq=None):
        if dq is None: dq = np.zeros(6, dtype=self.dtype)
        return np.array(self._grav_fun(*self._pack(q,dq)), dtype=self.dtype).reshape(6,)

    def coriolis(self, q, dq):
        return np.array(self._coriolis_fun(*self._pack(q,dq)), dtype=self.dtype).reshape(6,)

    def ddq(self, q, dq, Q):
        A = self.A(q, dq)
        h = self.h(q, dq)
        Q = np.asarray(Q, dtype=self.dtype).reshape(6,)
        return np.linalg.solve(A, Q - h)

    # ---- kinematics / contacts ----
    def keypoints(self, q, dq=None):
        if dq is None: dq = np.zeros(6, dtype=self.dtype)
        return np.array(self._keypoints_fun(*self._pack(q,dq)), dtype=self.dtype).reshape(6,2)

    def pToe(self, q, dq=None):
        if dq is None: dq = np.zeros(6, dtype=self.dtype)
        return np.array(self._pToe_fun(*self._pack(q,dq)), dtype=self.dtype).reshape(2,)

    def pHeel(self, q, dq=None):
        if dq is None: dq = np.zeros(6, dtype=self.dtype)
        return np.array(self._pHeel_fun(*self._pack(q,dq)), dtype=self.dtype).reshape(2,)

    def JToe(self, q, dq=None):
        if dq is None: dq = np.zeros(6, dtype=self.dtype)
        return np.array(self._JToe_fun(*self._pack(q,dq)), dtype=self.dtype).reshape(2,6)

    def JHeel(self, q, dq=None):
        if dq is None: dq = np.zeros(6, dtype=self.dtype)
        return np.array(self._JHeel_fun(*self._pack(q,dq)), dtype=self.dtype).reshape(2,6)

    def JToedot(self, q, dq):
        return np.array(self._JToedot_fun(*self._pack(q,dq)), dtype=self.dtype).reshape(2,6)

    def JHeeldot(self, q, dq):
        return np.array(self._JHeeldot_fun(*self._pack(q,dq)), dtype=self.dtype).reshape(2,6)

    def render_rgb(
        self,
        q,
        W=256, H=256,
        xlim=(-2.0, 2.0), ylim=(-2.2, 2.2),
        draw_ground=True,
        linewidth=3,
        square_color=(80, 180, 255),
        link_color=(240, 240, 240),
        foot_color=(200, 200, 200),
        bg_color=(15, 15, 18),
        ground_color=(90, 90, 90),
    ):
        """
        Return: uint8 image (H, W, 3).
        r0->rA drawn as a "square" centered at r0; vector r0->rA defines
        the center-to-edge-midpoint direction.
        """
        q = np.asarray(q, dtype=float).reshape(6,)
        P = np.asarray(self.keypoints(q), dtype=float)  # (6,2)
        r0, rA, rB, rC, rHeel, rToe = P

        # Auto bounds if not given
        if xlim is None or ylim is None:
            xs = P[:, 0]; ys = P[:, 1]
            cx = 0.5 * (xs.min() + xs.max())
            cy = 0.5 * (ys.min() + ys.max())
            span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1.0)
            pad = 0.6 * span
            if xlim is None: xlim = (cx - pad, cx + pad)
            if ylim is None: ylim = (cy - pad, cy + pad)

        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = np.array(bg_color, dtype=np.uint8)

        # Ground y=0
        if draw_ground:
            g0 = _world_to_pixel(np.array([[xlim[0], 0.0], [xlim[1], 0.0]]), W, H, xlim, ylim)
            (x0, y0), (x1, y1) = g0
            _draw_line(img, int(x0), int(y0), int(x1), int(y1), np.array(ground_color, np.uint8), thickness=max(1, linewidth//2))

        # Convert keypoints to pixel
        Ppx = _world_to_pixel(P, W, H, xlim, ylim)  # (6,2)
        p0, pA, pB, pC, pHeel, pToe = Ppx

        # Draw square centered at r0; direction defined by r0->rA
        u = (rA - r0).astype(float)
        L = float(np.linalg.norm(u))
        if L < 1e-9:
            uhat = np.array([1.0, 0.0])
        else:
            uhat = u / L
        nhat = np.array([-uhat[1], uhat[0]])

        # "center to edge midpoint" length: set to |r0->rA| by your description
        square_half = L

        # Build 4 corners in world
        c = r0
        corners_w = np.stack([
            c + square_half * uhat + square_half * nhat,
            c + square_half * uhat - square_half * nhat,
            c - square_half * uhat - square_half * nhat,
            c - square_half * uhat + square_half * nhat,
        ], axis=0)
        corners_px = _world_to_pixel(corners_w, W, H, xlim, ylim)

        _fill_convex_poly(img, corners_px, np.array(square_color, np.uint8))
        # outline
        for i in range(4):
            x0, y0 = corners_px[i]
            x1, y1 = corners_px[(i+1)%4]
            _draw_line(img, int(x0), int(y0), int(x1), int(y1), np.array(link_color, np.uint8), thickness=max(1, linewidth))

        # Draw links (r0->rA->rB->rC)
        pts = [p0, pA, pB, pC]
        for a, b in zip(pts[:-1], pts[1:]):
            _draw_line(img, int(a[0]), int(a[1]), int(b[0]), int(b[1]),
                       np.array(link_color, np.uint8), thickness=max(1, linewidth))

        # Draw foot segments
        _draw_line(img, int(pC[0]), int(pC[1]), int(pHeel[0]), int(pHeel[1]),
                   np.array(foot_color, np.uint8), thickness=max(1, linewidth))
        _draw_line(img, int(pC[0]), int(pC[1]), int(pToe[0]), int(pToe[1]),
                   np.array(foot_color, np.uint8), thickness=max(1, linewidth))

        return img

if __name__ == "__main__":
    params = dict(
    M=3.0, m1=1.5, m2=1.0, m3=2.0,
    I=0.01, I1=0.005, I2=0.005, I3=0.009,
    l0=0.3, l1=0.6, l2=0.6, l3=0.5,
    c1=0.3, c2=0.2, c3=0.4,
    g=9.81
    )

    dyn = LegDynamicsCached(params, cache_path= os.path.join(os.path.dirname(__file__), "leg3link_sympy_cache.pkl"), dtype=float)

    q  = np.array([0.0, 1.0, 0.0, 0.2, -0.4, 0.1])
    dq = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    print(dyn.A(q, dq))
    print(dyn.JToedot(q, dq))
    print(dyn.JHeeldot(q, dq))
