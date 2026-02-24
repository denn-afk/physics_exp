"""
Trajectory VAE for latent dynamics identification.

Design:
- Encoder: MLP over concatenated (s_t, a_t, s_{t+1}) transitions in a trajectory
           -> mean-pool over transitions -> output (mu_z, logvar_z)
- Decoder: MLP dynamics model p(s_{t+1} | s_t, a_t, z)
- Training: ELBO = E_q[log p(s_{t+1}|s_t,a_t,z)] - KL[q(z|tau) || p(z)]
- z_dim: 8 (let the model organize itself)
- Assumption: one z per trajectory, shared across all transitions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List, Tuple

# ---- import environment ----
from point_particle_env import PointParticleTrackEnv, DomainRandomizationCfg


# -----------------------------------------------------------------------
# Data collection
# -----------------------------------------------------------------------

def collect_trajectories(
    n_episodes: int = 500,
    seed: int = 42,
    kp: float = 20.0,
    kd: float = 8.0,
) -> List[dict]:
    """
    Run PD controller, collect transitions per episode.
    Returns list of dicts with keys:
        transitions: (T, obs_dim + act_dim + obs_dim) array of (s,a,s') 
        z_true: (mass, damping)
    """
    env = PointParticleTrackEnv(seed=seed)
    episodes = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        z_true = info["z_true"]  # (mass, damping), not fed to model

        transitions = []  # list of (s, a, s')
        s = obs.copy()

        for _ in range(env.horizon):
            px, py, vx, vy, ptx, pty, vtx, vty, _ = obs
            p = np.array([px, py])
            v = np.array([vx, vy])
            p_star = np.array([ptx, pty])
            v_star = np.array([vtx, vty])

            a = kp * (p_star - p) + kd * (v_star - v)
            a = np.clip(a, -env.a_max, env.a_max).astype(np.float32)

            obs_next, _, terminated, truncated, _ = env.step(a)
            s_next = obs_next.copy()

            transitions.append((s, a, s_next))
            s = s_next
            obs = obs_next

            if terminated or truncated:
                break

        episodes.append({
            "transitions": transitions,  # list of (s, a, s') tuples
            "z_true": z_true,
        })

    env.close()
    return episodes


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """
    Each item is one trajectory (episode).
    Returns:
        transitions: (T, s_dim + a_dim + s_dim) float32 tensor
        z_true: (2,) float32 tensor for evaluation only
    """

    def __init__(self, episodes: List[dict]):
        self.episodes = episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        trans = ep["transitions"]

        # stack (s, a, s') for each step
        rows = []
        for (s, a, s_next) in trans:
            rows.append(np.concatenate([s, a, s_next], axis=0))
        transitions = torch.tensor(np.stack(rows, axis=0), dtype=torch.float32)

        z_true = torch.tensor(ep["z_true"], dtype=torch.float32)
        return transitions, z_true


def collate_fn(batch):
    """Pad variable-length trajectories to same length."""
    transitions_list, z_true_list = zip(*batch)
    max_len = max(t.shape[0] for t in transitions_list)
    trans_dim = transitions_list[0].shape[1]

    padded = torch.zeros(len(batch), max_len, trans_dim)
    masks = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, t in enumerate(transitions_list):
        T = t.shape[0]
        padded[i, :T] = t
        masks[i, :T] = True

    z_true = torch.stack(z_true_list, dim=0)
    return padded, masks, z_true


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

class TransitionEncoder(nn.Module):
    """
    Encodes a trajectory of (s, a, s') transitions into q(z|tau).
    Architecture: MLP per transition -> mean pool -> MLP head -> (mu, logvar)
    """

    def __init__(self, trans_dim: int, z_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.trans_net = nn.Sequential(
            nn.Linear(trans_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, z_dim * 2),  # mu and logvar
        )
        self.z_dim = z_dim

    def forward(self, transitions: torch.Tensor, mask: torch.Tensor):
        """
        transitions: (B, T, trans_dim)
        mask: (B, T) bool, True = valid
        Returns: mu (B, z_dim), logvar (B, z_dim)
        """
        B, T, D = transitions.shape
        h = self.trans_net(transitions.view(B * T, D)).view(B, T, -1)

        # masked mean pool
        mask_f = mask.float().unsqueeze(-1)  # (B, T, 1)
        h = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)  # (B, hidden)

        out = self.head(h)
        mu, logvar = out.chunk(2, dim=-1)
        logvar = logvar.clamp(-10, 2)  # numerical stability
        return mu, logvar


class DynamicsDecoder(nn.Module):
    """
    p(s_{t+1} | s_t, a_t, z) -- Gaussian with fixed variance.
    """

    def __init__(self, s_dim: int, a_dim: int, z_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim + z_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, s_dim),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor, z: torch.Tensor):
        """
        s: (..., s_dim), a: (..., a_dim), z: (..., z_dim)
        Returns: s_next_pred (..., s_dim)
        """
        inp = torch.cat([s, a, z], dim=-1)
        return self.net(inp)


class TrajectoryVAE(nn.Module):

    def __init__(self, s_dim: int, a_dim: int, z_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        trans_dim = s_dim + a_dim + s_dim
        self.encoder = TransitionEncoder(trans_dim, z_dim, hidden_dim)
        self.decoder = DynamicsDecoder(s_dim, a_dim, z_dim, hidden_dim)
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.z_dim = z_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, transitions: torch.Tensor, mask: torch.Tensor):
        """
        transitions: (B, T, s+a+s)
        mask: (B, T)

        Returns: loss components
        """
        B, T, D = transitions.shape
        s_dim, a_dim = self.s_dim, self.a_dim

        # Encode
        mu, logvar = self.encoder(transitions, mask)
        z = self.reparameterize(mu, logvar)  # (B, z_dim)

        # Decode: for each transition, predict s_next
        s = transitions[:, :, :s_dim]               # (B, T, s_dim)
        a = transitions[:, :, s_dim:s_dim + a_dim]  # (B, T, a_dim)
        s_next_gt = transitions[:, :, s_dim + a_dim:]  # (B, T, s_dim)

        # Expand z to (B, T, z_dim)
        z_expand = z.unsqueeze(1).expand(-1, T, -1)

        s_next_pred = self.decoder(s, a, z_expand)  # (B, T, s_dim)

        # Reconstruction loss (MSE = negative log-likelihood with fixed variance)
        recon_loss = F.mse_loss(s_next_pred * mask.unsqueeze(-1),
                                s_next_gt * mask.unsqueeze(-1),
                                reduction='sum') / mask.sum().clamp(min=1)

        # KL divergence: KL[N(mu, sigma) || N(0,1)]
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        return recon_loss, kl_loss, mu, logvar, z


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_vae(
    model: TrajectoryVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 3e-4,
    beta: float = 1.0,  # KL weight (beta-VAE style)
    device: str = "cpu",
) -> dict:
    """Train the VAE and return loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_recon": [], "train_kl": [], "val_recon": [], "val_kl": []}
    model.to(device)

    for epoch in range(n_epochs):
        # --- train ---
        model.train()
        tr_recon, tr_kl = 0.0, 0.0
        for transitions, masks, _ in train_loader:
            transitions, masks = transitions.to(device), masks.to(device)
            recon, kl, *_ = model(transitions, masks)
            loss = recon + beta * kl

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_recon += recon.item()
            tr_kl += kl.item()

        scheduler.step()

        # --- val ---
        model.eval()
        vl_recon, vl_kl = 0.0, 0.0
        with torch.no_grad():
            for transitions, masks, _ in val_loader:
                transitions, masks = transitions.to(device), masks.to(device)
                recon, kl, *_ = model(transitions, masks)
                vl_recon += recon.item()
                vl_kl += kl.item()

        n_tr = len(train_loader)
        n_vl = len(val_loader)
        history["train_recon"].append(tr_recon / n_tr)
        history["train_kl"].append(tr_kl / n_tr)
        history["val_recon"].append(vl_recon / n_vl)
        history["val_kl"].append(vl_kl / n_vl)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"train recon={tr_recon/n_tr:.4f} kl={tr_kl/n_tr:.4f} | "
                  f"val recon={vl_recon/n_vl:.4f} kl={vl_kl/n_vl:.4f}")

    return history


# -----------------------------------------------------------------------
# Evaluation: does mu_z capture z_true?
# -----------------------------------------------------------------------

@torch.no_grad()
def evaluate_latent(model: TrajectoryVAE, loader: DataLoader, device: str = "cpu"):
    """
    Collect (mu_z, z_true) pairs and visualize.
    Goal: check if mu_z correlates with (mass, damping).
    """
    model.eval()
    all_mu, all_z_true = [], []

    for transitions, masks, z_true in loader:
        transitions, masks = transitions.to(device), masks.to(device)
        _, _, mu, _, _ = model(transitions, masks)
        all_mu.append(mu.cpu())
        all_z_true.append(z_true)

    all_mu = torch.cat(all_mu, dim=0).numpy()       # (N, z_dim)
    all_z_true = torch.cat(all_z_true, dim=0).numpy()  # (N, 2): mass, damping

    # Simple linear regression: can z_true be predicted from mu_z?
    from numpy.linalg import lstsq
    A = np.concatenate([all_mu, np.ones((len(all_mu), 1))], axis=1)
    for i, name in enumerate(["mass", "damping"]):
        coef, _, _, _ = lstsq(A, all_z_true[:, i], rcond=None)
        pred = A @ coef
        corr = np.corrcoef(pred, all_z_true[:, i])[0, 1]
        print(f"Linear prediction R for {name}: {corr:.3f}")

    # Scatter plot: color by mass and damping
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PCA of mu_z for visualization
    from numpy.linalg import svd
    mu_centered = all_mu - all_mu.mean(0)
    U, S, Vt = svd(mu_centered, full_matrices=False)
    mu_pca = mu_centered @ Vt[:2].T  # (N, 2)

    for ax, (param_idx, param_name) in zip(axes, [(0, "mass"), (1, "damping")]):
        sc = ax.scatter(mu_pca[:, 0], mu_pca[:, 1],
                        c=all_z_true[:, param_idx], cmap="viridis", alpha=0.6, s=20)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"Latent PCA colored by {param_name}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig("latent_evaluation.png", dpi=150)
    plt.close()
    print("Saved latent evaluation plot.")

    return all_mu, all_z_true


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Collect data
    print("Collecting trajectories...")
    episodes = collect_trajectories(n_episodes=1000, seed=0)

    # Train/val split
    split = int(0.8 * len(episodes))
    train_eps, val_eps = episodes[:split], episodes[split:]

    train_ds = TrajectoryDataset(train_eps)
    val_ds = TrajectoryDataset(val_eps)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 2. Build model
    # obs_dim=9, act_dim=2
    model = TrajectoryVAE(s_dim=9, a_dim=2, z_dim=8, hidden_dim=128)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Train
    print("Training VAE...")
    history = train_vae(model, train_loader, val_loader,
                        n_epochs=100, lr=3e-4, beta=0.001, device=device)

    # 4. Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(axes,
                               [("train_recon", "val_recon"), ("train_kl", "val_kl")],
                               ["Reconstruction Loss", "KL Loss"]):
        ax.plot(history[key[0]], label="train")
        ax.plot(history[key[1]], label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()

    # 5. Evaluate latent structure
    print("\nEvaluating latent space...")
    evaluate_latent(model, val_loader, device=device)

    # 6. Save model
    torch.save(model.state_dict(), "vae_model.pt")
    print("Done. Model saved.")
