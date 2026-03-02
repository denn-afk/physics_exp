import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class RecursiveEncoder(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, latent_dim=4, hidden_dim=128):
        super().__init__()
        # 输入：s_t, a_t, s_{t+1} (当前证据) + prev_mu, prev_logvar (历史先验)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        input_dim = state_dim + action_dim + state_dim + (latent_dim * 2)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 输出层：mu 和 logvar
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # 刚才讨论的约束：给 mu 加个 Tanh 缩放，防止跑飞
        self.mu_scale = 5.0 

    def forward(self, s, a, s_next, prev_mu, prev_logvar):
        # 拼接所有信息
        x = torch.cat([s, a, s_next, prev_mu, prev_logvar], dim=-1)
        h = self.network(x)
        
        mu = self.mu_scale * torch.tanh(self.mu_layer(h))
        logvar = self.logvar_layer(h) # logvar 范围不需要太死，由 KL 自动约束
        
        return mu, logvar
    
class TransitionDecoder(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, latent_dim=4, hidden_dim=128):
        super().__init__()
        # 输入：s_t, a_t (当前动作) + z_t (当前的隐变量判断)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        input_dim = state_dim + action_dim + latent_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim) # 输出预测的 s_{t+1} 或 Delta_s
        )

    def forward(self, s, a, z):
        x = torch.cat([s, a, z], dim=-1)
        detla_s =  self.network(x)
        return detla_s + s
    
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def gaussian_kl_divergence(mu_t, logvar_t, mu_prev, logvar_prev):
    """
    计算 KL(q_t || q_{t-1})
    mu_t, logvar_t: 当前步的分布参数 [batch, latent_dim]
    mu_prev, logvar_prev: 上一步的分布参数 (建议已 detach) [batch, latent_dim]
    """
    # 将 log_var 转回方差，方便看公式
    var_t = torch.exp(logvar_t)
    var_prev = torch.exp(logvar_prev)
    
    # 按照公式计算
    # 第一项: log(sigma_prev / sigma_t) = 0.5 * (logvar_prev - logvar_t)
    term1 = 0.5 * (logvar_prev - logvar_t)
    
    # 第二项: (var_t + (mu_t - mu_prev)^2) / (2 * var_prev)
    term2 = (var_t + (mu_t - mu_prev)**2) / (2 * var_prev)
    
    # 第三项: - 0.5
    kl = term1 + term2 - 0.5
    
    # 对隐变量维度求和，对 batch 取平均
    return kl.sum(dim=-1).mean()

# 核心训练步：处理一条轨迹
def step_and_learn(encoder:RecursiveEncoder, decoder:TransitionDecoder, optimizer, trajectory, beta=1.0, device='cpu'):
    T = trajectory['obs'].shape[0]
    
    # 凌晨三点的初始化：一片空白的 Prior
    prev_mu = torch.zeros(1, encoder.latent_dim).to(device)
    prev_logvar = torch.zeros(1, encoder.latent_dim).to(device)
    
    all_mu = [] # 记录下来，待会儿画图看
    optimizer.zero_grad()
    
    total_recon = 0
    total_kl = 0

    for t in range(T):
        # 1. 提取当前数据
        s, a, sn = trajectory['obs'][t:t+1], trajectory['act'][t:t+1], trajectory['next_obs'][t:t+1]
        s, a, sn = s.to(device), a.to(device), sn.to(device)

        # 2. 推断当前后验 (这一步，我们通过 detach 切断了和过去的联系)
        mu_t, logvar_t = encoder(s, a, sn, prev_mu.detach(), prev_logvar.detach())
        
        # 3. 采样 (这就是让你觉得怪的地方，它其实是给梯度加点扰动)
        z_t = reparameterize(mu_t, logvar_t)
        
        # 4. 验证并计算 Loss
        sn_pred = decoder(s, a, z_t)
        recon = F.mse_loss(sn_pred, sn)
        kl = gaussian_kl_divergence(mu_t, logvar_t, prev_mu.detach(), prev_logvar.detach())
        
        total_recon += recon
        total_kl += kl
        
        # 5. 传递接力棒
        prev_mu, prev_logvar = mu_t, logvar_t
        all_mu.append(mu_t.detach().cpu())

    # 梯度更新：让网络学会“如何更新先验”
    recon_loss = total_recon / T
    kl_loss = total_kl / T
    # loss = (total_recon + 0.1 * total_kl) / T
    loss = recon_loss + beta * kl_loss
    loss.backward()
    optimizer.step()
    
    return loss.item(), recon_loss.item(), kl_loss.item(), torch.cat(all_mu)

from point_particle_env import PointParticleTrackEnv, collect_trajectories

def prepare_trajectory(raw_traj_item, device='cpu'):
    """优化后的转换逻辑"""
    traj = raw_traj_item['traj']
    
    # 先用 np.stack 在 NumPy 层面合并成大块内存
    # 这样转 Tensor 时，PyTorch 只需要做一次高效的内存拷贝
    obs_np = np.stack([step['obs'] for step in traj])
    act_np = np.stack([step['action'] for step in traj])
    n_obs_np = np.stack([step['next_obs'] for step in traj])
    
    return {
        'obs': torch.from_numpy(obs_np).float().to(device),
        'act': torch.from_numpy(act_np).float().to(device),
        'next_obs': torch.from_numpy(n_obs_np).float().to(device),
        'z_true': raw_traj_item['z_true']
    }

if __name__ == "__main__": 
    # 实例化环境 (假设你昨天的类名为 PointParticleTrackEnv)
    env = PointParticleTrackEnv()
    # env.horizon = 2000  # 让它跑久一s点，看看长期的变化趋势

    print("开始采集数据...")
    raw_data = collect_trajectories(env, num_episodes=300)

    # raw_data[0]['z_true'] # 看看第一条轨迹的真实参数是什么样子的
    state_dim = len(raw_data[0]['traj'][0]['obs'])
    action_dim = len(raw_data[0]['traj'][0]['action'])
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")

    # 实例化模型
    latent_dim = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = RecursiveEncoder(state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim).to(device)
    decoder = TransitionDecoder(state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    # 记录loss变化，看看训练过程
    losses = []
    recon_losses = []    
    kl_losses = []

    print("开始训练...")
    for epoch in range(10):
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        pbar = tqdm(raw_data)
        for item in pbar:
            trajectory = prepare_trajectory(item, device)
            loss_val, recon_loss, kl_loss,mu_trace = step_and_learn(encoder, decoder, optimizer, trajectory, beta=100.0, device=device)
            epoch_loss += loss_val
            epoch_recon += recon_loss
            epoch_kl += kl_loss

            losses.append(loss_val)
            recon_losses.append(recon_loss)
            kl_losses.append(kl_loss)   

            pbar.set_description(f"Epoch {epoch}, Loss: {loss_val:.6f}, Recon: {recon_loss:.6f}, KL: {kl_loss:.6f}")
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Avg Loss: {epoch_loss/len(raw_data):.6f}, Recon Loss: {epoch_recon/len(raw_data):.6f}, KL Loss: {epoch_kl/len(raw_data):.6f}")

    # 训练完后，看一下最后一条轨迹的 latent 变化
    print("训练完成！最后一条轨迹的 Mu 均值:", mu_trace[-1].numpy())
    torch.save(encoder.state_dict(), "recursive_encoder.pth")
    torch.save(decoder.state_dict(), "transition_decoder.pth")

    np.savez("training_losses.npz", losses=losses, recon_losses=recon_losses, kl_losses=kl_losses)