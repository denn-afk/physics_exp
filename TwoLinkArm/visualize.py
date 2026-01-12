# realtime_twolink_sim_cv2.py
import time
import math
import cv2
import numpy as np
import torch

from fd_layer_taugrad import FDLayerQddTauGrad
from policy import TauPolicy  # 或把 TauPolicy 单独放 tau_policy.py 更干净


# -----------------------------
# Rendering helpers
# -----------------------------
def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)

def world_to_px(p: np.ndarray, origin_px: tuple[int,int], scale: float) -> tuple[int,int]:
    x_px = int(origin_px[0] + scale * p[0])
    y_px = int(origin_px[1] - scale * p[1])
    return x_px, y_px

def draw_text(img, text, x, y, color=(240,240,240), scale=0.55, thick=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def joint_positions(q: np.ndarray, l1: float, l2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q1, q2 = float(q[0]), float(q[1])
    p0 = np.array([0.0, 0.0], dtype=np.float64)
    p1 = np.array([l1*math.cos(q1), l1*math.sin(q1)], dtype=np.float64)
    p2 = p1 + np.array([l2*math.cos(q1+q2), l2*math.sin(q1+q2)], dtype=np.float64)
    return p0, p1, p2


# -----------------------------
# Angle wrapping (avoid OOD q)
# -----------------------------
def wrap_angle(theta: float) -> float:
    # map to (-pi, pi]
    return (theta + math.pi) % (2.0 * math.pi) - math.pi

def wrap_q_inplace(q: np.ndarray):
    q[0] = wrap_angle(float(q[0]))
    q[1] = wrap_angle(float(q[1]))


# -----------------------------
# Reference velocity generators
# -----------------------------
def dq_ref_constant(t: float) -> np.ndarray:
    return np.array([0.8, -0.2], dtype=np.float64)

def dq_ref_sine(t: float) -> np.ndarray:
    return np.array([
        0.8 + 0.25*math.sin(2.0*math.pi*0.5*t),
        -0.2 + 0.15*math.cos(2.0*math.pi*0.5*t),
    ], dtype=np.float64)


# -----------------------------
# Main simulation
# -----------------------------
def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")

    # ---- must match training params ----
    l1 = 1.0
    l2 = 1.0
    c1 = 0.5
    c2 = 0.5
    m1 = 3.0
    m2 = 2.0
    I1 = 0.2
    I2 = 0.15
    g  = 9.81

    # ---- load policy ----
    policy = TauPolicy(hidden=64).to(device)
    policy.load_state_dict(torch.load("policy_velocity_tracking.pth", map_location=device))
    policy.eval()

    # ---- dynamics layer ----
    dyn_layer = FDLayerQddTauGrad(
        l1=l1, l2=l2,
        c1=c1, c2=c2,
        m1=m1, m2=m2,
        I1=I1, I2=I2,
        g=g,
    ).to(device)

    # ---- sim params ----
    dt_sim = 0.002
    steps_per_frame = 10

    # torque limits
    use_tau_limit = True
    tau_limit = np.array([80.0, 80.0], dtype=np.float64)

    # (optional) dq limits to keep within training-ish range
    use_dq_limit = True
    dq_limit = np.array([6.0, 6.0], dtype=np.float64)

    # ---- initial state ----
    q  = np.array([0.3, -0.2], dtype=np.float64)
    dq = np.array([0.0,  0.0], dtype=np.float64)
    wrap_q_inplace(q)

    # ---- UI / render params ----
    W, H = 900, 700
    origin = (W//2, int(H*0.62))
    scale = 220.0

    mode = 0  # 0=constant, 1=sine
    paused = False

    window_name = "TwoLink realtime sim (policy torque control)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    t0 = time.time()
    last = time.time()
    fps_smooth = 60.0

    while True:
        now = time.time()

        # --- input handling ---
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord(' '):
            paused = not paused
        if key == ord('m'):
            mode = 1 - mode
        if key == ord('r'):
            q  = np.array([0.3, -0.2], dtype=np.float64)
            dq = np.array([0.0,  0.0], dtype=np.float64)
            wrap_q_inplace(q)

        if not paused:
            sim_t = now - t0
            dq_ref = dq_ref_constant(sim_t) if mode == 0 else dq_ref_sine(sim_t)

            for _ in range(steps_per_frame):
                q_t     = torch.as_tensor(q.reshape(1,2), device=device)
                dq_t    = torch.as_tensor(dq.reshape(1,2), device=device)
                dqref_t = torch.as_tensor(dq_ref.reshape(1,2), device=device)

                tau_t = policy(q_t, dq_t, dqref_t)
                tau = tau_t.detach().cpu().numpy().reshape(2)

                if use_tau_limit:
                    tau = np.clip(tau, -tau_limit, tau_limit)

                qdd_t = dyn_layer(q_t, dq_t, torch.as_tensor(tau.reshape(1,2), device=device))
                qdd = qdd_t.detach().cpu().numpy().reshape(2)

                # semi-implicit Euler
                dq = dq + dt_sim * qdd
                if use_dq_limit:
                    dq = np.clip(dq, -dq_limit, dq_limit)

                q  = q  + dt_sim * dq
                wrap_q_inplace(q)  # <<< keep q in (-pi, pi] to avoid OOD

        # --- render ---
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (14, 14, 14)

        p0, p1, p2 = joint_positions(q, l1, l2)
        p0px = world_to_px(p0, origin, scale)
        p1px = world_to_px(p1, origin, scale)
        p2px = world_to_px(p2, origin, scale)

        cv2.line(img, (0, origin[1]), (W, origin[1]), (50, 50, 50), 1)
        cv2.line(img, p0px, p1px, (90, 220, 120), 6, cv2.LINE_AA)
        cv2.line(img, p1px, p2px, (200, 140, 90), 6, cv2.LINE_AA)
        cv2.circle(img, p0px, 8, (230, 230, 230), -1, cv2.LINE_AA)
        cv2.circle(img, p1px, 8, (230, 230, 230), -1, cv2.LINE_AA)
        cv2.circle(img, p2px, 8, (230, 230, 230), -1, cv2.LINE_AA)

        # display reference
        sim_t = now - t0
        dq_ref = dq_ref_constant(sim_t) if mode == 0 else dq_ref_sine(sim_t)

        # compute tau/qdd for display
        q_t     = torch.as_tensor(q.reshape(1,2), device=device)
        dq_t    = torch.as_tensor(dq.reshape(1,2), device=device)
        dqref_t = torch.as_tensor(dq_ref.reshape(1,2), device=device)
        tau_t = policy(q_t, dq_t, dqref_t)
        tau = tau_t.detach().cpu().numpy().reshape(2)
        if use_tau_limit:
            tau = np.clip(tau, -tau_limit, tau_limit)
        qdd_t = dyn_layer(q_t, dq_t, torch.as_tensor(tau.reshape(1,2), device=device))
        qdd = qdd_t.detach().cpu().numpy().reshape(2)

        # FPS
        dt_real = now - last
        last = now
        if dt_real > 1e-6:
            fps = 1.0 / dt_real
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps

        # overlay
        draw_text(img, "Keys: [space]=pause  [m]=toggle ref  [r]=reset  [q/esc]=quit", 18, 28, (210,210,210), 0.55, 1)
        draw_text(img, f"ref_mode: {'constant' if mode==0 else 'sine'}   paused: {paused}   FPS~{fps_smooth:.1f}", 18, 52, (210,210,210), 0.55, 1)

        draw_text(img, f"q     = [{q[0]: .3f}, {q[1]: .3f}] rad", 18, 92)
        draw_text(img, f"dq    = [{dq[0]: .3f}, {dq[1]: .3f}] rad/s", 18, 116)
        draw_text(img, f"dqref = [{dq_ref[0]: .3f}, {dq_ref[1]: .3f}] rad/s", 18, 140, (120, 200, 240))
        draw_text(img, f"qdd   = [{qdd[0]: .3f}, {qdd[1]: .3f}] rad/s^2", 18, 164, (200, 200, 120))
        draw_text(img, f"tau   = [{tau[0]: .3f}, {tau[1]: .3f}]", 18, 188, (200, 140, 90))

        err = dq - dq_ref
        draw_text(img, f"vel_err = [{err[0]: .3e}, {err[1]: .3e}]", 18, 212, (240, 160, 160))

        cv2.imshow(window_name, img)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()