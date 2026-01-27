import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from legged_dynamics import LegDynamicsCached
from sim import physics_step, wrap_angles


# ============================================================
# 1. Normalizer：输入输出都先做标准化（非常重要）
# ============================================================

class Normalizer:
    def __init__(self, data: np.ndarray):
        """
        data: (N, D)
        """
        self.mean = data.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = data.std(axis=0, keepdims=True).astype(np.float32)
        self.std[self.std < 1e-6] = 1e-6

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def decode(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean


# ============================================================
# 2. ResMLP 动力学网络：输入 [q, dq, tau]，输出 Δ[q, dq]
#    这样可以更好地学局部残差，精度通常比直接 output next_state 好
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        h = F.silu(self.fc1(x))
        h = self.fc2(h)
        return self.ln(x + h)  # residual + LN 稳一点


class DynNet(nn.Module):
    def __init__(self, in_dim=18, out_dim=12, hidden_dim=256, num_blocks=4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        """
        x: (B, in_dim) = [q, dq, tau] 标准化后的
        返回: (B, out_dim) = Δ[q, dq] 标准化后的
        """
        h = F.silu(self.fc_in(x))
        for blk in self.blocks:
            h = blk(h)
        y = self.fc_out(h)
        return y


# ============================================================
# 3. 数据采集：用你的 physics_step 跑一大堆 random torque
#    学的是 f(q, dq, tau) → (q_next, dq_next)
# ============================================================

def collect_dyn_dataset(
    dyn,
    num_steps=200_000,
    dt=0.01,
    tau_scale=40.0,
    reset_q=None,
    reset_dq=None,
):
    """
    返回:
        X: (N, 18) = [q, dq, tau]
        Y: (N, 12) = [q_next, dq_next]
    """
    if reset_q is None:
        reset_q = np.array([0.0, 2.0, 0.0, 0.3, -0.6, 0.2], dtype=np.float64)
    if reset_dq is None:
        reset_dq = np.zeros(6, dtype=np.float64)

    q = reset_q.copy()
    dq = reset_dq.copy()

    X_list = []
    Y_list = []

    for k in range(num_steps):
        # 随机力矩分布（你后面可以换成某个 policy）
        tau = np.random.uniform(-tau_scale, tau_scale, size=(6,))

        # one step of true physics
        q_next, dq_next, _ = physics_step(dyn, q, dq, dt, Q=tau)

        x = np.concatenate([q, dq, tau], axis=0)          # 18
        y = np.concatenate([q_next, dq_next], axis=0)     # 12
        X_list.append(x)
        Y_list.append(y)

        q, dq = q_next, dq_next

        # 简单防炸/重置：状态太离谱就重置
        if (np.abs(q).max() > 10.0) or (q[1] < -0.5) or (np.abs(dq).max() > 200.0):
            q = reset_q.copy()
            dq = reset_dq.copy()

        if (k + 1) % 10_000 == 0:
            print(f"[collect] {k+1}/{num_steps} steps")

    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.float32)
    print("[collect] dataset shapes:", X.shape, Y.shape)
    return X, Y


# ============================================================
# 4. 训练动力学网络：学 Δstate
# ============================================================

def train_dyn_net(
    X_np: np.ndarray,
    Y_np: np.ndarray,
    hidden_dim=256,
    num_blocks=4,
    batch_size=2048,
    num_epochs=200,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    X_np: (N, 18) = [q, dq, tau]
    Y_np: (N, 12) = [q_next, dq_next]
    网络实际学的是 ΔY = Y - current_state[:12]
    """

    # 当前状态 = [q, dq]
    S_curr = X_np[:, :12]          # (N, 12)
    dY_np = Y_np - S_curr          # Δ[q, dq]

    # 标准化
    x_norm = Normalizer(X_np)
    y_norm = Normalizer(dY_np)

    Xn = x_norm.encode(X_np)
    Yn = y_norm.encode(dY_np)

    dataset = TensorDataset(
        torch.from_numpy(Xn),
        torch.from_numpy(Yn),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device(device)
    net = DynNet(
        in_dim=X_np.shape[1],
        out_dim=dY_np.shape[1],
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    net.train()
    N = len(dataset)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = net(xb)
                loss = F.mse_loss(pred, yb)

            scaler.scale(loss).backward()
            # 防止梯度爆炸
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / N
        print(f"[epoch {epoch}] train MSE = {avg_loss:.6e}")

    return net, x_norm, y_norm


# ============================================================
# 5. 用学到的 dyn net 做 rollout，对比真物理
# ============================================================

@torch.no_grad()
def rollout_compare(
    dyn,
    net,
    x_norm,
    y_norm,
    steps=1000,
    dt=0.01,
    tau_scale=40.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    用同一串 tau，分别用:
        - 真正 physics_step
        - 学到的 DynNet
    rollout 一段时间，看误差和视觉效果。
    """
    device = torch.device(device)

    # 初始状态
    q_true  = np.array([0.0, 2.0, 0.0, 0.3, -0.6, 0.2], dtype=np.float32)
    dq_true = np.zeros(6, dtype=np.float32)

    q_pred = q_true.copy()
    dq_pred = dq_true.copy()

    traj_true = []
    traj_pred = []

    for k in range(steps):
        tau = np.random.uniform(-tau_scale, tau_scale, size=(6,)).astype(np.float32)

        # ---- 真物理 ----
        q_true, dq_true, _ = physics_step(dyn, q_true, dq_true, dt, Q=tau)

        # ---- 学的物理 ----
        x = np.concatenate([q_pred, dq_pred, tau], axis=0)[None, :]  # (1, 18)
        x_n = x_norm.encode(x)                                       # 标准化
        x_t = torch.from_numpy(x_n).to(device)
        dY_pred_n = net(x_t)                                         # (1, 12) 标准化的 Δstate
        dY_pred = y_norm.decode(dY_pred_n.cpu().numpy())[0]          # 还原 Δstate

        # 当前 state_pred = [q_pred, dq_pred]
        S_curr = np.concatenate([q_pred, dq_pred], axis=0)
        S_next = S_curr + dY_pred
        q_pred = S_next[:6]
        dq_pred = S_next[6:]

        # wrap 一下角度
        q_true = wrap_angles(q_true)
        q_pred = wrap_angles(q_pred)

        traj_true.append(np.concatenate([q_true, dq_true], axis=0))
        traj_pred.append(np.concatenate([q_pred, dq_pred], axis=0))

        if (k + 1) % 100 == 0:
            err = np.linalg.norm(q_true - q_pred)
            print(f"[rollout step {k+1}] |q_true - q_pred| = {err:.4e}")

        # 防止爆炸
        if np.abs(q_true).max() > 20 or np.abs(q_pred).max() > 20:
            print("[rollout] break: q too large")
            break

    traj_true = np.stack(traj_true)   # (T, 12)
    traj_pred = np.stack(traj_pred)

    return traj_true, traj_pred


# ============================================================
# 6. main：一键跑起来
# ============================================================

if __name__ == "__main__":
    # 1) 构建动力学
    params = dict(
        M=3.0, m1=1.5, m2=1.0, m3=2.0,
        I=0.01, I1=0.005, I2=0.005, I3=0.009,
        l0=0.3, l1=0.6, l2=0.6, l3=0.5,
        c1=0.3, c2=0.2, c3=0.4,
        g=9.81
    )
    dyn = LegDynamicsCached(params)

    # 2) 采数据（可以根据机器性能调小 num_steps，比如先 50k 看看）
    X, Y = collect_dyn_dataset(
        dyn,
        num_steps=200_000,
        dt=0.01,
        tau_scale=40.0,
        reset_q=np.array([0.0, 2.0, 0.0, 0.3, -0.6, 0.2]),
        reset_dq=np.zeros(6),
    )

    # 3) 训练网络
    net, x_norm, y_norm = train_dyn_net(
        X, Y,
        hidden_dim=512,     # 稍微大一点
        num_blocks=6,       # 堆多一点层
        batch_size=4096,
        num_epochs=300,     # 你可以多训一点
        lr=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    torch.save({
        "model": net.state_dict(),
        "x_mean": x_norm.mean,
        "x_std": x_norm.std,
        "y_mean": y_norm.mean,
        "y_std": y_norm.std,
    }, "dyn_net.pth")
    print("[save] dyn_net.pth saved.")

    # 4) 简单 rollout 对比一下
    traj_true, traj_pred = rollout_compare(
        dyn,
        net,
        x_norm,
        y_norm,
        steps=1000,
        dt=0.01,
        tau_scale=40.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 5) 可选：画一画某个关节的对比
    try:
        import matplotlib.pyplot as plt
        t = np.arange(traj_true.shape[0]) * 0.01
        idx = 1  # 看 q[1]，大概是高度

        plt.figure()
        plt.plot(t, traj_true[:, idx], label="true q[1]")
        plt.plot(t, traj_pred[:, idx], "--", label="pred q[1]")
        plt.legend()
        plt.xlabel("t [s]")
        plt.ylabel("q[1]")
        plt.title("True vs Learned dynamics (q[1])")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("matplotlib plot skipped:", e)
