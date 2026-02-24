"""
MNIST VAE with latent space visualization.

Two key visualizations:
1. Latent scatter: encode test set -> plot mu_z colored by digit class
2. Latent traversal: sample a 2D grid in latent space -> decode -> show generated digits

Run: python mnist_vae.py
Outputs: latent_scatter.png, latent_traversal.png
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, z_dim: int = 2):
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
    def __init__(self, z_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class VAE(nn.Module):
    def __init__(self, z_dim: int = 2):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.z_dim = z_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)

        # ELBO
        recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        return recon, mu, logvar, recon_loss, kl_loss


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def train(z_dim=2, n_epochs=20, beta=1.0, device='cpu'):
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    model = VAE(z_dim=z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

        n = len(train_loader)
        print(f"Epoch {epoch+1:2d} | recon={total_recon/n:.2f}  kl={total_kl/n:.2f}")

    return model, test_loader


# -----------------------------------------------------------------------
# Visualization 1: Latent scatter colored by digit class
# -----------------------------------------------------------------------

@torch.no_grad()
def plot_latent_scatter(model, test_loader, device='cpu'):
    model.eval()
    all_mu, all_labels = [], []

    for x, y in test_loader:
        mu, _ = model.encoder(x.to(device))
        all_mu.append(mu.cpu())
        all_labels.append(y)

    all_mu = torch.cat(all_mu).numpy()
    all_labels = torch.cat(all_labels).numpy()

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap('tab10')

    for digit in range(10):
        mask = all_labels == digit
        ax.scatter(all_mu[mask, 0], all_mu[mask, 1],
                   c=[cmap(digit)], label=str(digit),
                   alpha=0.4, s=8)

    ax.legend(title="Digit", markerscale=3, loc='upper right')
    ax.set_title("Latent space — test set colored by digit class")
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    plt.tight_layout()
    plt.savefig("latent_scatter.png", dpi=150)
    plt.close()
    print("Saved latent_scatter.png")


# -----------------------------------------------------------------------
# Visualization 2: Latent space traversal (2D grid -> decoded images)
# -----------------------------------------------------------------------

@torch.no_grad()
def plot_latent_traversal(model, device='cpu', grid_size=20, z_range=3.0):
    """
    Sample a grid of z values in [-z_range, z_range]^2,
    decode each, arrange into one big image.
    """
    model.eval()
    lin = np.linspace(-z_range, z_range, grid_size)
    canvas = np.zeros((28 * grid_size, 28 * grid_size))

    for i, z2 in enumerate(reversed(lin)):   # y-axis: z2 increases upward
        for j, z1 in enumerate(lin):          # x-axis: z1 increases rightward
            z = torch.tensor([[z1, z2]], dtype=torch.float32).to(device)
            img = model.decoder(z).cpu().squeeze().numpy()
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = img

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas, cmap='gray_r', origin='upper',
              extent=[-z_range, z_range, -z_range, z_range])
    ax.set_title("Latent space traversal — decoded digits")
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    plt.tight_layout()
    plt.savefig("latent_traversal.png", dpi=150)
    plt.close()
    print("Saved latent_traversal.png")


# -----------------------------------------------------------------------
# Visualization 3: Reconstruction comparison (real vs reconstructed)
# -----------------------------------------------------------------------

@torch.no_grad()
def plot_reconstructions(model, test_loader, device='cpu', n=10):
    model.eval()
    x, _ = next(iter(test_loader))
    x = x[:n].to(device)
    recon, *_ = model(x)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 2.8))
    for i in range(n):
        axes[0, i].imshow(x[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Recon", fontsize=9)
    plt.suptitle("Reconstructions", y=1.02)
    plt.tight_layout()
    plt.savefig("reconstructions.png", dpi=150)
    plt.close()
    print("Saved reconstructions.png")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # z_dim=2 so we can visualize directly
    model, test_loader = train(z_dim=2, n_epochs=20, beta=1.0, device=device)

    plot_latent_scatter(model, test_loader, device)
    plot_latent_traversal(model, device)
    plot_reconstructions(model, test_loader, device)

    print("\nAll done. Check: latent_scatter.png, latent_traversal.png, reconstructions.png")
