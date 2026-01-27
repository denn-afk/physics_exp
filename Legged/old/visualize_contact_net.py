import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from legged_dynamics import LegDynamicsCached
from sim import wrap_angles   # 还用你原来的 wrap_angles


# ============================================================
# 1) Δdq 网络：用 (q, dq_free, y_toe, y_heel) → 预测 Δdq
#    定义和你训练时保持完全一致
# ============================================================

class DQCorrectionNet(nn.Module):
    def __init__(self, n_dof, hidden_dim=128):
        super().__init__()
        self.n_dof = n_dof
        # 输入是 [q, dq_free, y_toe, y_heel]
        input_dim = 2 * n_dof + 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim - 1, hidden_dim),  # 去掉一个 DOF（base x）
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, n_dof),   # 输出 Δdq
        )

    def forward(self, q, dq_free, y_toe, y_heel):
        """
        q       : (B, n_dof)
        dq_free : (B, n_dof)
        y_toe   : (B, 1)
        y_heel  : (B, 1)
        返回:
            delta_dq: (B, n_dof)
        """
        x = torch.cat([q, dq_free, y_toe, y_heel], dim=-1)
        # 去掉 base x（q 的第 0 维），因为 wrap 后不连续且对 contact 不敏感
        x = x[:, 1:]
        delta_dq = self.mlp(x)
        return delta_dq


# ============================================================
# 2) 加载训练好的网络
# ============================================================

def load_dq_net(path="dq_correction_net.pth",
                n_dof=6,
                hidden_dim=256,
                device="cpu"):
    device = torch.device(device)
    net = DQCorrectionNet(n_dof=n_dof, hidden_dim=hidden_dim).to(device)
    state = torch.load(path, map_location=device)
    net.load_state_dict(state)
    net.eval()
    print(f"[load_dq_net] loaded weights from {path}")
    return net


# ============================================================
# 3) 用 network 版本的 contact 做一步 physics
# ============================================================

@torch.no_grad()
def physics_step_net(dyn,
                     net,
                     q,
                     dq,
                     dt=0.01,
                     mu=0.8,
                     beta=0.2,
                     iters=25,
                     device="cpu"):
    """
    用:
        free dynamics + Δdq_net
    替代原来的 PGS contact。
    这里为了跟你采数据时一致，ddq 里用的是 Q=0（无外力矩）。
    mu/beta/iters 在目前这个版本里用不到，只是保持接口一致方便以后替换。
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    dq = np.asarray(dq, dtype=np.float64).reshape(-1)
    n_dof = q.shape[0]

    # 1) free dynamics（和 collect_dq_dataset 一样，用 Q=0）
    Q = np.zeros_like(dq)
    ddq = dyn.ddq(q, dq, Q)
    dq_free = dq + dt * ddq

    # 2) contact 几何（在 free state 上）
    p_toe = dyn.pToe(q, dq_free)   # (2,)
    p_heel = dyn.pHeel(q, dq_free) # (2,)
    y_toe = float(p_toe[1])
    y_heel = float(p_heel[1])

    # 3) 调 net 预测 Δdq
    device = torch.device(device)
    q_t     = torch.tensor(q,       dtype=torch.float32, device=device).unsqueeze(0)    # (1, n_dof)
    dqf_t   = torch.tensor(dq_free, dtype=torch.float32, device=device).unsqueeze(0)   # (1, n_dof)
    ytoe_t  = torch.tensor([[y_toe]],  dtype=torch.float32, device=device)             # (1,1)
    yheel_t = torch.tensor([[y_heel]], dtype=torch.float32, device=device)             # (1,1)

    delta_dq_t = net(q_t, dqf_t, ytoe_t, yheel_t)                                      # (1, n_dof)
    delta_dq = delta_dq_t.squeeze(0).cpu().numpy()                                     # (n_dof,)

    dq_next = dq_free + delta_dq
    q_next = q + dt * dq_next
    q_next = wrap_angles(q_next)

    # 这里原来的 physics_step 返回了 contact，这里先用一个 dummy
    contact = {
        "y_toe": y_toe,
        "y_heel": y_heel,
        "delta_dq_norm": float(np.linalg.norm(delta_dq)),
    }
    return q_next, dq_next, contact


# ============================================================
# 4) 可视化 loop：用 network step 替代原来的 physics_step
# ============================================================

def loop_show_net(dyn,
                  net,
                  q,
                  dq,
                  dt=0.01,
                  mu=0.8,
                  beta=0.2,
                  iters=25,
                  W=480,
                  H=480,
                  render_num=2,
                  device="cpu"):
    counter = 0
    for step in range(10000):
        start_time = time.time()

        q, dq, contact = physics_step_net(
            dyn, net, q, dq,
            dt=dt, mu=mu, beta=beta, iters=iters, device=device
        )

        print(f"[step {step}] q = {q}, contact info = {contact}")

        # 简单安全条件，防炸
        # if q[1] < -0.2 or np.abs(q).max() > 10.0 or np.abs(dq).max() > 100.0:
        #     print("[loop_show_net] safety break triggered")
        #     break

        # 控制渲染频率（每 render_num 步渲染一次）
        if counter < render_num:
            counter += 1
        else:
            counter = 0
            t0 = time.time()
            rgb = dyn.render_rgb(q)              # 假设还是返回 RGB
            cv2.imshow("sim_net", rgb[:, :, ::-1])  # OpenCV 用 BGR
            if (cv2.waitKey(1) & 0xFF) == 27:   # ESC 退出
                break
            print("render time:", time.time() - t0)

        # 简单 real-time 控制
        elapsed = time.time() - start_time
        if elapsed < dt:
            time.sleep(dt - elapsed)

    cv2.destroyAllWindows()


# ============================================================
# 5) main：构建 dyn, load net, 跑可视化
# ============================================================

if __name__ == "__main__":
    # 初始状态（高度稍微高一点）
    q0  = np.array([0.0, 3.0, 0.0, 0.3, -0.6, 0.2])
    dq0 = np.zeros(6)

    params = dict(
        M=3.0, m1=1.5, m2=1.0, m3=2.0,
        I=0.01, I1=0.005, I2=0.005, I3=0.009,
        l0=0.3, l1=0.6, l2=0.6, l3=0.5,
        c1=0.3, c2=0.2, c3=0.4,
        g=9.81
    )
    dyn = LegDynamicsCached(params)

    # 加载训练好的 Δdq 网络
    net = load_dq_net(
        path="dq_correction_net.pth",
        n_dof=6,
        hidden_dim=256,   # 和你训练时设的一致
        device="cpu",
    )

    # 用 network contact 跑可视化
    loop_show_net(dyn, net, q0, dq0, dt=0.01, render_num=2, device="cpu")
