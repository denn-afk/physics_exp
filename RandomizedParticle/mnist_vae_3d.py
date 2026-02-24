"""
MNIST VAE with 3D latent space visualization.
z_dim=3, trained longer, interactive 3D scatter plot.

Run: python mnist_vae_3d.py
Output: latent_3d.html (interactive), latent_3d_static.png
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------
# Model (same as before, z_dim=3)
# -----------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, z_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, z_dim)
        self.logvar_head = nn.Linear(256, z_dim)

    def forward(self, x):
        h = self.net(x.view(x.size(0), -1))
        return self.mu_head(h), self.logvar_head(h).clamp(-10, 2)


class Decoder(nn.Module):
    def __init__(self, z_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class VAE(nn.Module):
    def __init__(self, z_dim=3):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.z_dim = z_dim

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return recon, mu, logvar, recon_loss, kl_loss


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train(z_dim=3, n_epochs=40, beta=1.0, device='cpu'):
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)

    model = VAE(z_dim=z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        model.train()
        total_recon, total_kl = 0.0, 0.0
        for x, _ in train_loader:
            x = x.to(device)
            _, _, _, recon_loss, kl_loss = model(x)
            loss = recon_loss + beta * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        scheduler.step()

        n = len(train_loader)
        print(f"Epoch {epoch+1:2d}/{n_epochs} | "
              f"recon={total_recon/n:.2f}  kl={total_kl/n:.2f}")

    return model, test_loader


# -----------------------------------------------------------------------
# Collect latents
# -----------------------------------------------------------------------

@torch.no_grad()
def collect_latents(model, test_loader, device='cpu'):
    model.eval()
    all_mu, all_labels = [], []
    for x, y in test_loader:
        mu, _ = model.encoder(x.to(device))
        all_mu.append(mu.cpu())
        all_labels.append(y)
    return torch.cat(all_mu).numpy(), torch.cat(all_labels).numpy()


# -----------------------------------------------------------------------
# Visualization 1: Interactive 3D scatter (plotly -> HTML)
# -----------------------------------------------------------------------

def plot_3d_interactive(mu, labels):
    try:
        import plotly.graph_objects as go

        colors = [
            '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
            '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'
        ]

        fig = go.Figure()
        for digit in range(10):
            mask = labels == digit
            fig.add_trace(go.Scatter3d(
                x=mu[mask, 0], y=mu[mask, 1], z=mu[mask, 2],
                mode='markers',
                marker=dict(size=2, color=colors[digit], opacity=0.5),
                name=str(digit),
            ))

        fig.update_layout(
            title="MNIST 3D Latent Space (z_dim=3)",
            scene=dict(
                xaxis_title="z₁",
                yaxis_title="z₂",
                zaxis_title="z₃",
            ),
            legend_title="Digit",
            width=900, height=750,
        )

        fig.write_html("latent_3d.html")
        print("Saved interactive plot: latent_3d.html  (open in browser)")

    except ImportError:
        print("plotly not installed, skipping interactive plot. pip install plotly")


# -----------------------------------------------------------------------
# Visualization 2: Static 3D scatter (matplotlib, 4 viewpoints)
# -----------------------------------------------------------------------

def plot_3d_static(mu, labels):
    cmap = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(14, 12))

    # 4 different camera angles
    viewpoints = [
        (30, 45,  "View 1"),
        (30, 135, "View 2"),
        (30, 225, "View 3"),
        (60, 45,  "Top-down"),
    ]

    for idx, (elev, azim, title) in enumerate(viewpoints):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

        for digit in range(10):
            mask = labels == digit
            # subsample for speed
            idx_sample = np.where(mask)[0]
            if len(idx_sample) > 300:
                idx_sample = np.random.choice(idx_sample, 300, replace=False)
            ax.scatter(
                mu[idx_sample, 0],
                mu[idx_sample, 1],
                mu[idx_sample, 2],
                c=[cmap(digit)],
                label=str(digit),
                alpha=0.5, s=6,
            )

        ax.set_title(title)
        ax.set_xlabel("z₁", fontsize=8)
        ax.set_ylabel("z₂", fontsize=8)
        ax.set_zlabel("z₃", fontsize=8)
        ax.view_init(elev=elev, azim=azim)

        if idx == 0:
            ax.legend(title="Digit", markerscale=3,
                      fontsize=7, loc='upper left', ncol=2)

    plt.suptitle("MNIST 3D Latent Space — 4 viewpoints", fontsize=14)
    plt.tight_layout()
    plt.savefig("latent_3d_static.png", dpi=150)
    plt.close()
    print("Saved static plot: latent_3d_static.png")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, test_loader = train(z_dim=3, n_epochs=40, beta=1.0, device=device)

    print("\nCollecting latents...")
    mu, labels = collect_latents(model, test_loader, device)

    plot_3d_interactive(mu, labels)   # needs plotly
    plot_3d_static(mu, labels)        # always works

    print("\nDone.")
