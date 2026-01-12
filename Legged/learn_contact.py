# train_residual.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from legged_dynamics import LegDynamicsCached
from leg_env import LegEnv


# ---------- 1) 一个很傻但安全的 PD policy ----------

def pd_stand_policy(obs, noise_std=2.0):
    """
    obs: [q(6), dq(6)]
    输出一个 torque，用来大致保持站立 + 小噪声。
    """
    q = obs[:6]
    dq = obs[6:]

    # 参考姿态：脚大概在地上，身体稍微前倾一点
    q_ref = np.array([0.0, 1.8, 0.0, 0.3, -0.6, 0.3], dtype=float)

    # 只对旋转关节做 PD，浮动基不直接加 torque
    Kp = np.array([0.0, 0.0, 50.0, 60.0, 60.0, 40.0], dtype=float)
    Kd = np.array([0.0, 0.0,  5.0,  5.0,  5.0,  3.0], dtype=float)

    tau_pd = Kp * (q_ref - q) - Kd * dq

    noise = noise_std * np.random.randn(6)
    tau = tau_pd + noise
    return tau


# ---------- 2) rollout 收 residual 数据 ----------

def collect_residual_dataset(
    dyn,
    num_episodes=200,
    horizon=300,
    dt=0.01,
    noise_std=2.0,
):
    """
    返回:
      X: [N, 18]  (q, dq, tau)
      Y: [N, 6]   residual = M ddq_real + h - tau
    """
    env = LegEnv(dyn, dt=dt)
    X = []
    Y = []

    for ep in tqdm(range(num_episodes), desc="Collecting dataset"):
        obs = env.reset(randomize=True)

        for t in range(horizon):
            # 1) 用当前观测选 action
            tau = pd_stand_policy(obs, noise_std=noise_std)

            # 记录 step 前的状态
            q = env.q.copy()
            dq = env.dq.copy()

            # 2) 环境前进一步
            obs_next, done, info = env.step(tau)

            # 3) 用实际速度变化估 ddq_real
            dq_next = env.dq.copy()
            ddq_real = (dq_next - dq) / env.dt

            # 4) 计算 residual = M ddq_real + h - tau
            M = dyn.A(q, dq)
            h = dyn.h(q, dq)
            residual = M @ ddq_real + h - tau

            # （可选 safety）把特别离谱的样本干掉，防止数值爆掉训练
            if not np.isfinite(residual).all():
                if done:
                    break
                else:
                    obs = obs_next
                    continue

            X.append(np.concatenate([q, dq, tau], axis=0))
            Y.append(residual)

            obs = obs_next
            if done:
                break

    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)

    print("Collected dataset:", X.shape, Y.shape)
    print("Residual stats: mean |r| =", np.mean(np.linalg.norm(Y, axis=1)),
          "max |r| =", np.max(np.linalg.norm(Y, axis=1)))
    return X, Y


# ---------- 3) 一个小 MLP 学 residual(q,dq,tau) ----------

class ResidualNet(nn.Module):
    def __init__(self, in_dim=18, out_dim=6, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_residual_net(X, Y,
                       epochs=50,
                       batch_size=512,
                       lr=1e-3,
                       device="cpu"):

    X_t = torch.from_numpy(X).float().to(device)
    Y_t = torch.from_numpy(Y).float().to(device)

    model = ResidualNet(in_dim=X.shape[1], out_dim=Y.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    N = X_t.shape[0]
    num_batches = max(1, N // batch_size)

    for ep in range(1, epochs+1):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0

        for i in range(num_batches):
            idx = perm[i*batch_size:(i+1)*batch_size]
            xb = X_t[idx]
            yb = Y_t[idx]

            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        epoch_loss /= num_batches
        if ep % 5 == 0 or ep == 1:
            print(f"[Epoch {ep:03d}] train MSE: {epoch_loss:.6f}")

    return model


# ---------- 4) main: 收数据 + 训练 + 打印 loss ----------

if __name__ == "__main__":
    # 物理参数（和你 main 里的一样）
    params = dict(
        M=3.0, m1=1.5, m2=1.0, m3=2.0,
        I=0.01, I1=0.005, I2=0.005, I3=0.009,
        l0=0.3, l1=0.6, l2=0.6, l3=0.5,
        c1=0.3, c2=0.2, c3=0.4,
        g=9.81
    )
    dyn = LegDynamicsCached(params)

    # 1) rollout 收一批数据
    print("Collecting residual dataset...")
    X, Y = collect_residual_dataset(
        dyn,
        num_episodes=200,    # 可以先减成 20 跑快一点
        horizon=300,
        dt=0.01,
        noise_std=2.0,
    )

    # 2) 训练 residual 网络，观察 loss
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_residual_net(
        X, Y,
        epochs=2000,
        batch_size=512,
        lr=1e-3,
        device=device
    )

    # 可选：随便测几条的误差
    with torch.no_grad():
        X_sample = torch.from_numpy(X[:10]).float().to(device)
        Y_sample = torch.from_numpy(Y[:10]).float().to(device)
        Y_hat = model(X_sample)
        err = (Y_hat - Y_sample).cpu().numpy()
        print("Sample residual |error|:", np.linalg.norm(err, axis=1))
