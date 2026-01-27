import numpy as np
from legged_dynamics import LegDynamicsCached
import torch
import torch.nn as nn
import torch.nn.functional as F
from sim import physics_step

class ContactNet(nn.Module):
    def __init__(self, n_dof, hidden_dim=128):
        super().__init__()
        self.n_dof = n_dof

        # 这里随便选个输入维数：q, dq_free, v_toe, v_heel, y_toe, y_heel
        # q: n_dof, dq_free: n_dof,
        # v_toe: 2, v_heel: 2, y_toe: 1, y_heel: 1
        input_dim = 2 * n_dof + 2 + 2 + 1 + 1

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            # 输出 4 维: [ln_toe, lt_toe, ln_heel, lt_heel]
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, q, dq_free, v_toe_free, v_heel_free,
                y_toe, y_heel, mu):
        """
        输入:
            q           : (B, n_dof)
            dq_free     : (B, n_dof)
            v_toe_free  : (B, 2)  [vx, vy]
            v_heel_free : (B, 2)
            y_toe       : (B, 1)
            y_heel      : (B, 1)
            mu          : float 或 tensor (可 broadcast)

        输出:
            impulses: dict
                {
                  "toe":  (lt_toe, ln_toe),
                  "heel": (lt_heel, ln_heel)
                }
            每个 lt / ln 形状: (B, 1)
        """
        # 拼 feature
        x = torch.cat(
            [q, dq_free,
             v_toe_free, v_heel_free,
             y_toe, y_heel],
            dim=-1
        )  # (B, input_dim)

        raw = self.mlp(x)  # (B, 4)
        ln_toe_raw, lt_toe_raw, ln_heel_raw, lt_heel_raw = torch.chunk(raw, 4, dim=-1)

        # ---- 约束 lambda_n >= 0 ----
        # 用 softplus: 保证 ln >= 0 且可导
        ln_toe = F.softplus(ln_toe_raw)
        ln_heel = F.softplus(ln_heel_raw)

        # ---- 约束 |lambda_t| <= mu * lambda_n ----
        # 先给一个“方向” lt_raw，再用 tanh 限幅
        # lt = μ * ln * tanh(lt_raw)
        # 这样天然保证 |lt| <= μ ln
        if not torch.is_tensor(mu):
            mu = torch.tensor(float(mu), device=x.device, dtype=x.dtype)

        lt_toe = mu * ln_toe * torch.tanh(lt_toe_raw)
        lt_heel = mu * ln_heel * torch.tanh(lt_heel_raw)

        impulses = {
            "toe":  {"lt": lt_toe,  "ln": ln_toe},
            "heel": {"lt": lt_heel, "ln": ln_heel},
        }
        return impulses


def desired_contact_velocity(y, v_free, beta, dt):
    phi_slop = 1e-4  # 和 resolve_toe_heel_pgs 里一致的小容忍
    """
    根据你的 PGS 激活规则，给出目标 post-impact 速度。
    - 输入:
        y      : 接触点当前高度 (相对地面 y=0)
        v_free : pre-impact 速度 [vx, vy]
    - 输出:
        v_des  : 期望的 post-impact 速度 [vx_des, vy_des]
    """
    vx, vy = float(v_free[0]), float(v_free[1])

    # pre-impact 法向速度就是 vy（因为 n = [0,1]）
    vn = vy

    # activation + bn rule:
    #   if y < 0.0:        bn = max(0, beta * (-y) / dt)
    #   elif y <= phi_slop and vn < 0: bn = 0
    #   else: inactive
    if y < 0.0:
        bn = max(0.0, beta * (-y) / max(dt, 1e-12))
    elif (y <= phi_slop) and (vn < 0.0):
        bn = 0.0
    else:
        # inactive: desired = free
        return v_free.copy()

    # active contact: 切向目标速度设为 0（sticking preference），
    # 法向设为 bn
    vt_des = 0.0
    vn_des = bn
    return np.array([vt_des, vn_des], dtype=float)


def train_step_contactnet(
    dyn: LegDynamicsCached,
    net: ContactNet,
    optimizer,
    q_np,
    dq_np,
    dt=0.01,
    mu=0.8,
    beta=0.2,
    device="cpu"
):
    """
    对一个 (q, dq) 做一次训练 step：
    ContactNet 输出 toe/heel 冲量 λ = (lt, ln)，
    通过 M^{-1} J^T 施加到 dq_free 上，得到 dq_plus，
    再算 toe/heel 的 v_plus，让它靠近 desired_contact_velocity。
    """
    net.train()

    # ---- numpy 输入整理 ----
    q = np.asarray(q_np, float).reshape(-1)
    dq = np.asarray(dq_np, float).reshape(-1)
    n_dof = q.shape[0]

    # ---- 1) free dynamics ----
    Q = np.zeros_like(dq)
    ddq_free = dyn.ddq(q, dq, Q)      # (n_dof,)
    dq_free = dq + dt * ddq_free      # (n_dof,)

    # ---- 2) contact Jacobians & positions ----
    J_toe_np = dyn.JToe(q, dq)        # (2, n_dof)
    J_heel_np = dyn.JHeel(q, dq)      # (2, n_dof)

    p_toe = dyn.pToe(q, dq)           # (2,)
    p_heel = dyn.pHeel(q, dq)         # (2,)
    y_toe = float(p_toe[1])
    y_heel = float(p_heel[1])

    # ---- 3) free contact velocities ----
    v_toe_free = J_toe_np @ dq_free   # (2,)
    v_heel_free = J_heel_np @ dq_free # (2,)

    # ---- 4) desired contact velocities ----
    v_toe_des_np = desired_contact_velocity(y_toe, v_toe_free, beta, dt)   # (2,)
    v_heel_des_np = desired_contact_velocity(y_heel, v_heel_free, beta, dt)

    # ---- 5) torch tensors ----
    device = torch.device(device)

    q_t = torch.tensor(q[None, :], dtype=torch.float32, device=device)             # (1, n_dof)
    dq_free_t = torch.tensor(dq_free[None, :], dtype=torch.float32, device=device)
    v_toe_free_t = torch.tensor(v_toe_free[None, :], dtype=torch.float32, device=device)   # (1, 2)
    v_heel_free_t = torch.tensor(v_heel_free[None, :], dtype=torch.float32, device=device)
    y_toe_t = torch.tensor([[y_toe]], dtype=torch.float32, device=device)          # (1, 1)
    y_heel_t = torch.tensor([[y_heel]], dtype=torch.float32, device=device)        # (1, 1)

    v_toe_des_t = torch.tensor(v_toe_des_np[None, :], dtype=torch.float32, device=device)   # (1, 2)
    v_heel_des_t = torch.tensor(v_heel_des_np[None, :], dtype=torch.float32, device=device)

    # J, Minv 当常量，不对其求导（这里直接用 tensor 就行）
    J_toe_t = torch.tensor(J_toe_np, dtype=torch.float32, device=device)           # (2, n_dof)
    J_heel_t = torch.tensor(J_heel_np, dtype=torch.float32, device=device)
    M_np = dyn.A(q, dq)                                                            # (n_dof, n_dof)
    Minv_np = np.linalg.inv(M_np)
    Minv_t = torch.tensor(Minv_np, dtype=torch.float32, device=device)             # (n_dof, n_dof)

    # ---- 6) ContactNet 输出 toe/heel 冲量 λ = (lt, ln) ----
    impulses = net(
        q_t,
        dq_free_t,
        v_toe_free_t,
        v_heel_free_t,
        y_toe_t,
        y_heel_t,
        mu
    )
    lt_toe = impulses["toe"]["lt"]    # (1, 1)
    ln_toe = impulses["toe"]["ln"]    # (1, 1)
    lt_heel = impulses["heel"]["lt"]  # (1, 1)
    ln_heel = impulses["heel"]["ln"]  # (1, 1)

    # 组成 world-frame impulse 向量 [ft, fn]
    lam_toe = torch.cat([lt_toe, ln_toe], dim=-1)     # (1, 2)
    lam_heel = torch.cat([lt_heel, ln_heel], dim=-1)  # (1, 2)

    # ---- 7) generalized impulse τ_imp = J^T λ ----
    # (1,2) @ (2,n_dof) -> (1,n_dof)
    tau_imp_toe = (J_toe_t.T @ lam_toe.T).T               # (1, n_dof)
    tau_imp_heel = (J_heel_t.T @ lam_heel.T).T             # (1, n_dof)
    tau_imp = tau_imp_toe + tau_imp_heel              # (1, n_dof)

    # ---- 8) post-impact dq_plus ----
    # 列向量公式是 dq_plus = dq_free + Minv @ τ_imp^T
    # 行向量写法是 dq_plus = dq_free + τ_imp @ Minv^T
    dq_free_row = dq_free_t                           # (1, n_dof)
    dq_plus = dq_free_row + tau_imp @ Minv_t.T        # (1, n_dof)

    # ---- 9) post-impact contact velocities ----
    # v_plus_row = dq_plus @ J^T
    v_toe_plus = dq_plus @ J_toe_t.T                  # (1, 2)
    v_heel_plus = dq_plus @ J_heel_t.T                # (1, 2)

    # ---- 10) loss: v_plus 靠近 v_des ----
    loss_toe = torch.mean((v_toe_plus - v_toe_des_t) ** 2)
    loss_heel = torch.mean((v_heel_plus - v_heel_des_t) ** 2)
    loss = loss_toe + loss_heel

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())

params = dict(
    M=3.0, m1=1.5, m2=1.0, m3=2.0,
    I=0.01, I1=0.005, I2=0.005, I3=0.009,
    l0=0.3, l1=0.6, l2=0.6, l3=0.5,
    c1=0.3, c2=0.2, c3=0.4,
    g=9.81
)
dyn = LegDynamicsCached(params)

n_dof = 6
net = ContactNet(n_dof=n_dof, hidden_dim=512)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
while True:
    q = np.array([0.0, 2.0, 0.0, 0.3, -0.6, 0.2])
    dq = np.zeros(n_dof)
    for step in range(10000):
        q, dq, contact = physics_step(dyn, q, dq, 0.01, mu=0.8, beta=0.2, iters=20)

        loss = train_step_contactnet(
            dyn, net, optimizer,
            q, dq,
            dt=0.01, mu=0.8, beta=0.2, device="cpu"
        )

        print("loss:", loss)


