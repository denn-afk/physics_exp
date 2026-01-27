import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from legged_dynamics import LegDynamicsCached
from contact_pgs import resolve_toe_heel_pgs
from sim import wrap_angles   # 你自己那个 [-pi,pi] 的 wrap


# -------------------------------
# 1) Δdq 网络：用 (q, dq_free, y_toe, y_heel) → 预测 Δdq
# -------------------------------

class DQCorrectionNet(nn.Module):
    def __init__(self, n_dof, hidden_dim=128):
        super().__init__()
        self.n_dof = n_dof
        # [q, dq_free, y_toe, y_heel]
        input_dim = 2 * n_dof + 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim-1, hidden_dim),
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
        x = x[:, 1:]  # 去掉 x 位置（因为 wrap 后不连续）
        delta_dq = self.mlp(x)
        return delta_dq


# -------------------------------
# 2) 单步采样：从 PGS 得到一条 (input, Δdq) 配对
# -------------------------------

def sample_one_dq_pair(dyn, q, dq, dt, mu=0.8, beta=0.2, iters=20):
    """
    给定当前 (q,dq)，通过 PGS 走一步，
    返回用于训练 Δdq-net 的一条样本：
        inputs: q, dq_free, y_toe, y_heel
        target: delta_dq = dq_plus - dq_free
    """
    q = np.asarray(q, float).reshape(-1)
    dq = np.asarray(dq, float).reshape(-1)
    n_dof = q.shape[0]

    Q = np.zeros_like(dq)

    # 1) free dynamics
    ddq = dyn.ddq(q, dq, Q)
    dq_free = dq + dt * ddq

    # 2) contact geometry at free state
    p_toe = dyn.pToe(q, dq_free)   # (2,)
    p_heel = dyn.pHeel(q, dq_free)
    y_toe = float(p_toe[1])
    y_heel = float(p_heel[1])

    # 3) PGS contact
    dq_plus, contact = resolve_toe_heel_pgs(
        dyn, q, dq_free, dt,
        mu=mu, beta=beta, iters=iters
    )

    # 4) target Δdq
    delta_dq = dq_plus - dq_free

    # 打包成训练用 feature/label
    x = {
        # 注意：这里存的是“当前这步”的 q / dq_free / y
        "q": q.copy(),
        "dq_free": dq_free.copy(),
        "y_toe": y_toe,
        "y_heel": y_heel,
    }
    y = delta_dq
    return x, y


# -------------------------------
# 3) Rollout 多条轨迹，采集 dataset
# -------------------------------

def collect_dq_dataset(
    dyn,
    num_episodes=50,
    steps_per_episode=200,
    dt=0.01,
    mu=0.8,
    beta=0.2,
    iters=20,
):
    """
    返回:
        X_q      : (N, n_dof)
        X_dqf    : (N, n_dof)
        X_ytoe   : (N, 1)
        X_yheel  : (N, 1)
        Y_delta  : (N, n_dof)
    """
    X_list = []
    Y_list = []

    q0  = np.array([0.0, 2.0, 0.0, 0.3, -0.6, 0.2])
    dq0 = np.zeros(6)

    for ep in range(num_episodes):
        # 初始状态加一点扰动
        q = q0 + np.random.randn(6) * np.array([0.05, 0.1, 0.05, 0.1, 0.1, 0.1])
        dq = dq0 + np.random.randn(6) * 0.2
        q = wrap_angles(q)

        for t in range(steps_per_episode):
            x, delta_dq = sample_one_dq_pair(dyn, q, dq, dt, mu=mu, beta=beta, iters=iters)

            # 更新状态（真正下一步用的是 dq_plus）
            dq_plus = x["dq_free"] + delta_dq
            q = q + dt * dq_plus
            dq = dq_plus
            q = wrap_angles(q)

            # 只在有非零 contact 的时候存（可选）
            if np.linalg.norm(delta_dq) > 1e-5:
                X_list.append(x)
                Y_list.append(delta_dq)

            # 简单安全条件
            if q[1] < -0.2 or np.abs(q).max() > 10.0 or np.abs(dq).max() > 100.0:
                break

        print(f"[collect] ep {ep}, samples so far = {len(X_list)}")

    # ---- 整理成 numpy ----
    n_dof = q0.shape[0]
    N = len(X_list)
    X_q = np.zeros((N, n_dof), dtype=np.float32)
    X_dqf = np.zeros((N, n_dof), dtype=np.float32)
    X_ytoe = np.zeros((N, 1), dtype=np.float32)
    X_yheel = np.zeros((N, 1), dtype=np.float32)
    Y = np.zeros((N, n_dof), dtype=np.float32)

    for i, (x, delta_dq) in enumerate(zip(X_list, Y_list)):
        X_q[i] = x["q"]
        X_dqf[i] = x["dq_free"]
        X_ytoe[i, 0] = x["y_toe"]
        X_yheel[i, 0] = x["y_heel"]
        Y[i] = delta_dq

    return X_q, X_dqf, X_ytoe, X_yheel, Y

from torch.utils.data import TensorDataset, DataLoader

def train_dq_net(
    X_q_np,
    X_dqf_np,
    X_ytoe_np,
    X_yheel_np,
    Y_np,
    hidden_dim=128,
    batch_size=256,
    num_epochs=50,
    lr=1e-3,
    device="cpu",
):
    """
    用采集好的 (q, dq_free, y_toe, y_heel) → Δdq 数据训练 DQCorrectionNet
    """
    device = torch.device(device)

    X_q = torch.tensor(X_q_np, dtype=torch.float32)
    X_dqf = torch.tensor(X_dqf_np, dtype=torch.float32)
    X_ytoe = torch.tensor(X_ytoe_np, dtype=torch.float32)
    X_yheel = torch.tensor(X_yheel_np, dtype=torch.float32)
    Y = torch.tensor(Y_np, dtype=torch.float32)

    dataset = TensorDataset(X_q, X_dqf, X_ytoe, X_yheel, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_dof = X_q_np.shape[1]
    net = DQCorrectionNet(n_dof=n_dof, hidden_dim=hidden_dim).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_n = 0

        for q_b, dqf_b, ytoe_b, yheel_b, y_b in loader:
            q_b = q_b.to(device)
            dqf_b = dqf_b.to(device)
            ytoe_b = ytoe_b.to(device)
            yheel_b = yheel_b.to(device)
            y_b = y_b.to(device)

            delta_pred = net(q_b, dqf_b, ytoe_b, yheel_b)
            loss = F.mse_loss(delta_pred, y_b)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * q_b.size(0)
            total_n += q_b.size(0)

        avg_loss = total_loss / max(1, total_n)
        print(f"[epoch {epoch}] loss = {avg_loss:.6e}")

    return net

# 收数据
params = dict(
M=3.0, m1=1.5, m2=1.0, m3=2.0,
I=0.01, I1=0.005, I2=0.005, I3=0.009,
l0=0.3, l1=0.6, l2=0.6, l3=0.5,
c1=0.3, c2=0.2, c3=0.4,
g=9.81
)
dyn = LegDynamicsCached(params)
X_q, X_dqf, X_ytoe, X_yheel, Y = collect_dq_dataset(dyn,
                                                    num_episodes=1000,
                                                    steps_per_episode=1000,
                                                    dt=0.01,
                                                    mu=0.8, beta=0.2, iters=25)
print("dataset:", X_q.shape, Y.shape)
print(Y)
# 训练
net = train_dq_net(X_q, X_dqf, X_ytoe, X_yheel, Y,
                   hidden_dim=256,
                   batch_size=512,
                   num_epochs=2000,
                   lr=1e-3,
                   device="cpu")

torch.save(net.state_dict(), "dq_correction_net.pth")
print("Model saved to dq_correction_net.pth")