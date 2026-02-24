"""
Incremental Trajectory VAE for latent dynamics identification.

Key idea: encoder processes one transition (s, a, s') at a time.
Posterior q(z|tau_{1:t}) is updated incrementally via running mean pool.
This enables online Bayesian-style adaptation: as more transitions arrive,
the posterior narrows toward the true latent parameters.

At deploy time: maintain a running sum of transition embeddings,
update mu_z / sigma_z after each step.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List

from point_particle_env import PointParticleTrackEnv, DomainRandomizationCfg


# -----------------------------------------------------------------------
# Data collection (same as before)
# -----------------------------------------------------------------------

def collect_trajectories(n_episodes=500, seed=42, kp=20.0, kd=8.0):
    env = PointParticleTrackEnv(seed=seed)
    episodes = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        z_true = info["z_true"]  # (mass, damping) -- only used for evaluation

        transitions, s = [], obs.copy()
        for _ in range(env.horizon):
            px, py, vx, vy, ptx, pty, vtx, vty, _ = obs
            a = kp * (np.array([ptx, pty]) - np.array([px, py])) + \
                kd * (np.array([vtx, vty]) - np.array([vx, vy]))
            a = np.clip(a, -env.a_max, env.a_max).astype(np.float32)

            obs_next, _, terminated, truncated, _ = env.step(a)
            transitions.append((s, a, obs_next.copy()))
            s = obs_next.copy()
            obs = obs_next
            if terminated or truncated:
                break

        episodes.append({"transitions": transitions, "z_true": z_true})

    env.close()
    return episodes


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    def __init__(self, episodes):
        self.episodes = episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        rows = [np.concatenate([s, a, sn]) for s, a, sn in ep["transitions"]]
        transitions = torch.tensor(np.stack(rows), dtype=torch.float32)
        # transitions: (T, s_dim + a_dim + s_dim)
        z_true = torch.tensor(ep["z_true"], dtype=torch.float32)
        # z_true: (2,) -- mass and damping, for eval only
        return transitions, z_true


def collate_fn(batch):
    transitions_list, z_true_list = zip(*batch)
    max_len = max(t.shape[0] for t in transitions_list)
    trans_dim = transitions_list[0].shape[1]

    # padded: (B, T_max, trans_dim)
    padded = torch.zeros(len(batch), max_len, trans_dim)
    # mask: (B, T_max) -- True where valid
    masks = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, t in enumerate(transitions_list):
        T = t.shape[0]
        padded[i, :T] = t
        masks[i, :T] = True

    z_true = torch.stack(z_true_list)  # (B, 2)
    return padded, masks, z_true


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

class TransitionEmbedder(nn.Module):
    """
    Embeds a single transition (s, a, s') into a fixed-size vector.
    This is the core building block for incremental inference:
    each transition gets its own embedding, then they are mean-pooled.

    Input:  (*, trans_dim)  where trans_dim = s_dim + a_dim + s_dim
    Output: (*, hidden_dim)
    """
    def __init__(self, trans_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(trans_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )

    def forward(self, x):
        # x: (*, trans_dim) -> (*, hidden_dim)
        return self.net(x)


class PosteriorHead(nn.Module):
    """
    Maps a pooled embedding to posterior parameters (mu, logvar).

    Input:  (B, hidden_dim)  -- mean-pooled transition embeddings
    Output: mu     (B, z_dim)
            logvar (B, z_dim)
    """
    def __init__(self, hidden_dim=128, z_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, z_dim * 2),
        )
        self.z_dim = z_dim

    def forward(self, h):
        # h: (B, hidden_dim)
        out = self.net(h)               # (B, z_dim * 2)
        mu, logvar = out.chunk(2, dim=-1)  # each (B, z_dim)
        return mu, logvar.clamp(-10, 2)


class DynamicsDecoder(nn.Module):
    """
    Predicts next state given current state, action, and latent z.
    p(s_{t+1} | s_t, a_t, z)

    Input:  s (*, s_dim), a (*, a_dim), z (*, z_dim)
    Output: s_next_pred (*, s_dim)
    """
    def __init__(self, s_dim, a_dim, z_dim=8, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim + z_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, s_dim),
        )

    def forward(self, s, a, z):
        # s: (*, s_dim), a: (*, a_dim), z: (*, z_dim)
        return self.net(torch.cat([s, a, z], dim=-1))  # (*, s_dim)


class IncrementalVAE(nn.Module):
    """
    VAE with incremental posterior inference.

    The key design:
    1. Each transition (s, a, s') is independently embedded: e_t = embedder(s, a, s')
    2. Posterior is computed from mean-pooled embeddings: h = mean(e_1, ..., e_t)
    3. This means at deploy time, we can maintain a running mean and update
       the posterior after each new transition -- no need to re-encode the full history.

    Training uses the full trajectory, but the architecture is identical
    to incremental inference because mean-pool is order-independent and
    supports online updates: h_t = (h_{t-1} * (t-1) + e_t) / t
    """

    def __init__(self, s_dim, a_dim, z_dim=8, hidden_dim=128):
        super().__init__()
        trans_dim = s_dim + a_dim + s_dim
        self.embedder = TransitionEmbedder(trans_dim, hidden_dim)
        self.posterior_head = PosteriorHead(hidden_dim, z_dim)
        self.decoder = DynamicsDecoder(s_dim, a_dim, z_dim, hidden_dim)
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.z_dim = z_dim

    def encode(self, transitions, mask):
        """
        Encode a batch of (possibly padded) trajectories into posterior params.

        transitions: (B, T, trans_dim)
        mask:        (B, T)          -- True where valid
        returns:     mu (B, z_dim), logvar (B, z_dim)
        """
        B, T, D = transitions.shape

        # Embed each transition independently
        e = self.embedder(transitions.view(B * T, D))  # (B*T, hidden_dim)
        e = e.view(B, T, -1)                           # (B, T, hidden_dim)

        # Masked mean pool: aggregate over valid transitions only
        # mask_f: (B, T, 1) for broadcasting
        mask_f = mask.float().unsqueeze(-1)            # (B, T, 1)
        h = (e * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)  # (B, hidden_dim)

        return self.posterior_head(h)                  # mu, logvar each (B, z_dim)

    def reparameterize(self, mu, logvar):
        # mu, logvar: (B, z_dim)
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)  # (B, z_dim)

    def forward(self, transitions, mask):
        """
        Full forward pass for training.

        transitions: (B, T, trans_dim)
        mask:        (B, T)
        """
        B, T, _ = transitions.shape
        s_dim, a_dim = self.s_dim, self.a_dim

        # --- Encode full trajectory -> posterior ---
        mu, logvar = self.encode(transitions, mask)    # (B, z_dim)
        z = self.reparameterize(mu, logvar)            # (B, z_dim)

        # --- Decode: predict s_{t+1} for every transition ---
        s      = transitions[:, :, :s_dim]             # (B, T, s_dim)
        a      = transitions[:, :, s_dim:s_dim+a_dim]  # (B, T, a_dim)
        s_next = transitions[:, :, s_dim+a_dim:]       # (B, T, s_dim)

        # Expand z across time dimension for batched decoding
        z_exp = z.unsqueeze(1).expand(-1, T, -1)       # (B, T, z_dim)

        s_next_pred = self.decoder(s, a, z_exp)        # (B, T, s_dim)

        # Reconstruction loss: MSE only over valid (unmasked) transitions
        # mask.unsqueeze(-1): (B, T, 1) broadcast over s_dim
        recon = F.mse_loss(
            s_next_pred * mask.unsqueeze(-1),
            s_next      * mask.unsqueeze(-1),
            reduction='sum'
        ) / mask.sum().clamp(min=1)

        # KL divergence: KL[ N(mu, sigma) || N(0, I) ]
        # = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        return recon, kl, mu, logvar, z

    # ------------------------------------------------------------------
    # Incremental inference API (for deploy / online adaptation)
    # ------------------------------------------------------------------

    def new_context(self):
        """
        Initialize an empty context for a new episode.
        Returns a dict holding the running state for incremental inference.
        """
        return {
            "running_sum": None,  # sum of transition embeddings so far
            "count": 0,           # number of transitions seen
        }

    @torch.no_grad()
    def update_context(self, context, s, a, s_next):
        """
        Incorporate one new transition into the posterior estimate.

        s, a, s_next: numpy arrays or 1D tensors (single transition)
        Returns: mu (z_dim,), sigma (z_dim,) -- updated posterior params
        """
        # Pack transition into tensor: (1, trans_dim)
        trans = torch.tensor(
            np.concatenate([s, a, s_next])[None], dtype=torch.float32
        )

        # Embed this single transition: (1, hidden_dim)
        e = self.embedder(trans)

        # Update running sum
        if context["running_sum"] is None:
            context["running_sum"] = e
        else:
            context["running_sum"] = context["running_sum"] + e
        context["count"] += 1

        # Compute current posterior from running mean
        h = context["running_sum"] / context["count"]  # (1, hidden_dim)
        mu, logvar = self.posterior_head(h)            # (1, z_dim)

        return mu.squeeze(0), torch.exp(0.5 * logvar).squeeze(0)  # (z_dim,), (z_dim,)


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train_vae(model, train_loader, val_loader,
              n_epochs=100, lr=3e-4, beta=0.01, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    history = {"train_recon": [], "train_kl": [], "val_recon": [], "val_kl": []}
    model.to(device)

    for epoch in range(n_epochs):
        model.train()
        tr_r, tr_k = 0.0, 0.0
        for transitions, masks, _ in train_loader:
            transitions, masks = transitions.to(device), masks.to(device)
            recon, kl, *_ = model(transitions, masks)
            loss = recon + beta * kl
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_r += recon.item(); tr_k += kl.item()
        scheduler.step()

        model.eval()
        vl_r, vl_k = 0.0, 0.0
        with torch.no_grad():
            for transitions, masks, _ in val_loader:
                transitions, masks = transitions.to(device), masks.to(device)
                recon, kl, *_ = model(transitions, masks)
                vl_r += recon.item(); vl_k += kl.item()

        n_tr, n_vl = len(train_loader), len(val_loader)
        for k, v in zip(["train_recon","train_kl","val_recon","val_kl"],
                         [tr_r/n_tr, tr_k/n_tr, vl_r/n_vl, vl_k/n_vl]):
            history[k].append(v)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"train recon={tr_r/n_tr:.4f} kl={tr_k/n_tr:.4f} | "
                  f"val recon={vl_r/n_vl:.4f} kl={vl_k/n_vl:.4f}")

    return history


# -----------------------------------------------------------------------
# Evaluation: posterior convergence over time
# -----------------------------------------------------------------------

@torch.no_grad()
def plot_posterior_convergence(model, episodes, n_episodes=5, device="cpu"):
    """
    For a few episodes, show how mu_z and sigma_z evolve as more
    transitions arrive. This visualizes the incremental Bayesian update:
    posterior should narrow toward z_true as t increases.
    """
    model.eval()
    fig, axes = plt.subplots(n_episodes, 2, figsize=(12, 3 * n_episodes))

    for ep_idx in range(n_episodes):
        ep = episodes[ep_idx]
        z_true = ep["z_true"]  # (mass, damping)

        context = model.new_context()
        mus, sigmas = [], []

        for s, a, s_next in ep["transitions"]:
            mu, sigma = model.update_context(context, s, a, s_next)
            # mu: (z_dim,), sigma: (z_dim,) -- posterior params after t transitions
            mus.append(mu.numpy())
            sigmas.append(sigma.numpy())

        mus    = np.array(mus)    # (T, z_dim)
        sigmas = np.array(sigmas) # (T, z_dim)
        T = len(mus)

        for ax, title in zip(axes[ep_idx], ["mu_z (mean)", "sigma_z (std)"]):
            data = mus if title.startswith("mu") else sigmas
            for dim in range(data.shape[1]):
                ax.plot(data[:, dim], alpha=0.5, linewidth=0.8)
            ax.set_title(f"Ep {ep_idx} | mass={z_true[0]:.2f} damp={z_true[1]:.2f} | {title}")
            ax.set_xlabel("transitions seen")
            # sigma should decrease over time -- posterior getting more certain
            if title.startswith("sigma"):
                ax.set_ylabel("uncertainty")

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/posterior_convergence.png", dpi=150)
    plt.close()
    print("Saved posterior_convergence.png")


@torch.no_grad()
def plot_latent_scatter(model, val_loader, device="cpu"):
    """PCA of mu_z colored by mass and damping."""
    model.eval()
    all_mu, all_z_true = [], []
    for transitions, masks, z_true in val_loader:
        transitions, masks = transitions.to(device), masks.to(device)
        mu, _ = model.encode(transitions, masks)
        all_mu.append(mu.cpu()); all_z_true.append(z_true)

    all_mu     = torch.cat(all_mu).numpy()      # (N, z_dim)
    all_z_true = torch.cat(all_z_true).numpy()  # (N, 2)

    # PCA to 2D for visualization
    mu_c = all_mu - all_mu.mean(0)
    _, _, Vt = np.linalg.svd(mu_c, full_matrices=False)
    mu_pca = mu_c @ Vt[:2].T  # (N, 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (i, name) in zip(axes, [(0, "mass"), (1, "damping")]):
        sc = ax.scatter(mu_pca[:, 0], mu_pca[:, 1],
                        c=all_z_true[:, i], cmap="viridis", alpha=0.6, s=20)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"Latent PCA colored by {name}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/latent_scatter.png", dpi=150)
    plt.close()
    print("Saved latent_scatter.png")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Collecting trajectories...")
    episodes = collect_trajectories(n_episodes=500, seed=0)
    split = int(0.8 * len(episodes))
    train_eps, val_eps = episodes[:split], episodes[split:]

    train_loader = DataLoader(TrajectoryDataset(train_eps), batch_size=32,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(TrajectoryDataset(val_eps),   batch_size=32,
                              shuffle=False, collate_fn=collate_fn)

    model = IncrementalVAE(s_dim=9, a_dim=2, z_dim=8, hidden_dim=128)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("Training...")
    train_vae(model, train_loader, val_loader,
              n_epochs=100, lr=3e-4, beta=0.01, device=device)

    print("\nEvaluating...")
    plot_latent_scatter(model, val_loader, device)
    plot_posterior_convergence(model, val_eps, n_episodes=5, device=device)

    torch.save(model.state_dict(), "/mnt/user-data/outputs/incremental_vae.pt")
    print("Done.")
