
from torch import nn
import torch


class TauPolicy(nn.Module):
    """
    Input: [q(2), dq(2), dq_ref(2)] -> tau(2)
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, q, dq, dq_ref):
        # q: (...,2)
        q_sin = torch.sin(q)
        q_cos = torch.cos(q)
        x = torch.cat([q_sin, q_cos, dq, dq_ref], dim=-1)  # (..., 8)
        return self.net(x)

