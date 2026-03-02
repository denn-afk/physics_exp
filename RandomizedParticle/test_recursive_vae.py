import torch
import matplotlib.pyplot as plt
import numpy as np
from point_particle_env import PointParticleTrackEnv, collect_trajectories
from recursive_vae import RecursiveEncoder, TransitionDecoder
from tqdm import tqdm

# --- 1. 这里的参数要和训练时对齐 ---
STATE_DIM = 9
ACTION_DIM = 2
LATENT_DIM = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 加载模型 ---
encoder = RecursiveEncoder(state_dim=STATE_DIM, action_dim=ACTION_DIM, latent_dim=LATENT_DIM).to(DEVICE)
# Decoder 虽然这次绘图不用，但 load 进来备用也是好的
decoder = TransitionDecoder(state_dim=STATE_DIM, action_dim=ACTION_DIM, latent_dim=LATENT_DIM).to(DEVICE)

encoder.load_state_dict(torch.load("recursive_encoder.pth", map_location=DEVICE))
decoder.load_state_dict(torch.load("transition_decoder.pth", map_location=DEVICE))
encoder.eval() 
print("模型加载成功！")

# --- 3. 获取一条新鲜的轨迹 ---
env = PointParticleTrackEnv()
env.horizon = 2000
raw_data = collect_trajectories(env, num_episodes=1) # 只拿一条
traj_item = raw_data[0]
z_true = traj_item['z_true']

# --- 4. 推断 Latent 变化过程 ---
def get_mu_trace(encoder, traj_item):
    traj = traj_item['traj']
    T = len(traj)
    
    # 初始化 Prior
    prev_mu = torch.zeros(1, LATENT_DIM).to(DEVICE)
    prev_logvar = torch.zeros(1, LATENT_DIM).to(DEVICE)
    
    mu_history = []
    
    with torch.no_grad():
        for step in traj:
            s = torch.tensor(step['obs'], dtype=torch.float32).view(1, -1).to(DEVICE)
            a = torch.tensor(step['action'], dtype=torch.float32).view(1, -1).to(DEVICE)
            sn = torch.tensor(step['next_obs'], dtype=torch.float32).view(1, -1).to(DEVICE)
            
            # 递归推断
            mu_t, logvar_t = encoder(s, a, sn, prev_mu, prev_logvar)
            
            mu_history.append(mu_t.cpu().numpy().flatten())
            # 更新下一轮的输入
            prev_mu, prev_logvar = mu_t, logvar_t
            
    return np.array(mu_history)

mu_trace = get_mu_trace(encoder, traj_item)

# --- 5. 可视化 ---
plt.figure(figsize=(12, 6))
steps = np.arange(len(mu_trace))

for i in range(LATENT_DIM):
    plt.plot(steps, mu_trace[:, i], label=f'Latent Dim {i}', lw=2)

plt.axvline(x=10, color='gray', linestyle='--', alpha=0.5, label='Early Adaptation')
plt.title(f"Latent Inference over Time\nTrue Params (Mass, Damping): {z_true}", fontsize=14)
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Value of Mu", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# print(f"该环境真实参数: {z_true}")
# print(f"收敛后的 Mu 均值: {mu_trace[-1]}")

# from sklearn.decomposition import PCA

# def run_latent_clustering_analysis(encoder, n_envs=50, steps_per_env=100):
#     """
#     随机生成不同环境，记录每个环境稳定后的 Latent 坐标。
#     """
#     encoder.eval()
#     all_final_mus = []
#     all_mass_values = []
#     test_env = PointParticleTrackEnv()

    
#     print(f"正在分析 {n_envs} 个随机环境...")
    
#     for _ in tqdm(range(n_envs)):
        
#         # 2. 运行一小段轨迹让模型进行“认知推断”
#         # 我们直接用随机动作来产生信息增益
#         prev_mu = torch.zeros(1, LATENT_DIM).to(DEVICE)
#         prev_logvar = torch.zeros(1, LATENT_DIM).to(DEVICE)
        
#         obs, _ = test_env.reset()
#         true_mass = test_env.mass

#         print(f"环境参数 (Mass): {true_mass:.3f}")

#         for _ in range(steps_per_env):
#             action = test_env.action_space.sample()
#             next_obs, _, _, _, _ = test_env.step(action)
            
#             s = torch.tensor(obs, dtype=torch.float32).view(1, -1).to(DEVICE)
#             a = torch.tensor(action, dtype=torch.float32).view(1, -1).to(DEVICE)
#             sn = torch.tensor(next_obs, dtype=torch.float32).view(1, -1).to(DEVICE)
            
#             with torch.no_grad():
#                 mu_t, logvar_t = encoder(s, a, sn, prev_mu, prev_logvar)
#                 prev_mu, prev_logvar = mu_t, logvar_t
#             obs = next_obs
            
#         # 3. 收集稳定后的 mu (代表模型对该环境的身份识别)
#         all_final_mus.append(prev_mu.cpu().numpy().flatten())
#         all_mass_values.append(true_mass)

#     # --- 开始降维与绘图 ---
#     mus_array = np.array(all_final_mus)
#     mass_array = np.array(all_mass_values)
    
#     # 使用 PCA 将 4D Latent 压到 2D 
#     pca = PCA(n_components=2)
#     mus_2d = pca.fit_transform(mus_array)
    
#     plt.figure(figsize=(10, 7))
#     # 颜色映射代表真实的 Mass
#     scatter = plt.scatter(mus_2d[:, 0], mus_2d[:, 1], c=mass_array, 
#                          cmap='plasma', s=100, alpha=0.8, edgecolors='k')
    
#     plt.colorbar(scatter, label='Physical Mass (True)')
#     plt.title("Latent Space Topology: How Model Maps Physical Mass", fontsize=14)
#     plt.xlabel("Latent Principal Component 1")
#     plt.ylabel("Latent Principal Component 2")
#     plt.grid(True, alpha=0.2)
#     plt.show()

# # 执行分析
# run_latent_clustering_analysis(encoder)