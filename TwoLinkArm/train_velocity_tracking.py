# train_velocity_tracking.py
import torch
import torch.nn as nn
import torch.optim as optim

from fd_layer_taugrad import FDLayerQddTauGrad
from policy import TauPolicy



def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- dynamics params (example) ----
    dyn_layer = FDLayerQddTauGrad(
        l1=1.0, l2=1.0,
        c1=0.5, c2=0.5,
        m1=3.0, m2=2.0,     # 你可以改大/改小观察梯度尺度
        I1=0.2, I2=0.15,
        g=9.81,
    ).to(device)

    policy = TauPolicy(hidden=64).to(device)
    opt = optim.Adam(policy.parameters(), lr=3e-3)

    dt = 0.01
    lam_tau = 1e-8  # torque regularization

    # ---- toy training data: random states, random desired velocities ----
    B = 256  # batch
    for it in range(10000):
        # random q, dq in some range
        q = (torch.rand(B, 2, device=device) * 2.0 - 1.0)*torch.pi  # [-pi,pi]
        dq = (torch.rand(B, 2, device=device) * 2.0 - 1.0)* torch.pi  # [-pi,pi]

        # desired velocity (can be random or a function of q)
        # dq_ref = torch.zeros(B, 2, device=device)
        # dq_ref[:, 0] = 0.8  # example: want joint1 vel ~0.8
        # dq_ref[:, 1] = -0.2 # and joint2 vel ~-0.2
        dq_ref = (torch.rand(B, 2, device=device) * 2.0 - 1.0)* torch.pi  # [-2pi,2pi]


        # ---- forward ----
        tau = policy(q, dq, dq_ref)                 # (B,2)
        qdd = dyn_layer(q, dq, tau)                 # (B,2), grad only flows to tau
        dq_next = dq + dt * qdd                     # (B,2)

        # ---- loss: single-step velocity tracking ----
        err = dq_next - dq_ref
        loss_track = 0.5 * (err * err).sum(dim=-1).mean()
        loss_reg = 0.5 * lam_tau * (tau * tau).sum(dim=-1).mean()
        loss = loss_track + loss_reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        opt.step()

        if it % 200 == 0:
            with torch.no_grad():
                # quick diagnostics
                grad_norm = 0.0
                for p in policy.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item()
                print(f"it={it:4d} loss={loss.item():.6f} track={loss_track.item():.6f} reg={loss_reg.item():.6f} grad_sum_norm={grad_norm:.3f}")

    # ---- sanity check on a sample ----
    with torch.no_grad():
        q = torch.tensor([[0.3, -0.2]], device=device)
        dq = torch.tensor([[0.1, 0.0]], device=device)
        dq_ref = torch.tensor([[0.8, -0.2]], device=device)
        tau = policy(q, dq, dq_ref)
        qdd = dyn_layer(q, dq, tau)
        dq_next = dq + dt * qdd
        print("\nSample:")
        print("tau     =", tau.cpu().numpy())
        print("qdd     =", qdd.cpu().numpy())
        print("dq_next =", dq_next.cpu().numpy(), "  target=", dq_ref.cpu().numpy())

        torch.save(policy.state_dict(), "policy_velocity_tracking.pth")


if __name__ == "__main__":
    main()
