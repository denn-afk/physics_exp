# fd_layer_taugrad.py
import numpy as np
import torch
import torch.nn as nn
from twolink_dynamics import TwoLinkDynamics


class _FDQddTauGradFn(torch.autograd.Function):
    """
    qdd = M(q)^{-1} (tau - h(q,dq))

    Backward ONLY returns grad w.r.t tau:
        grad_tau = solve(M(q)^T, grad_qdd)
    No grads for q, dq.
    """

    @staticmethod
    def forward(ctx, q: torch.Tensor, dq: torch.Tensor, tau: torch.Tensor, dyn, np_dtype):
        if q.shape[-1] != 2 or dq.shape[-1] != 2 or tau.shape[-1] != 2:
            raise ValueError("q, dq, tau must have last dimension = 2")

        device = q.device
        out_dtype = q.dtype

        # flatten batch dims
        q_np = q.detach().cpu().numpy().astype(np_dtype, copy=False).reshape(-1, 2)
        dq_np = dq.detach().cpu().numpy().astype(np_dtype, copy=False).reshape(-1, 2)
        tau_np = tau.detach().cpu().numpy().astype(np_dtype, copy=False).reshape(-1, 2)

        B = q_np.shape[0]
        qdd_np = np.zeros((B, 2), dtype=np_dtype)
        M_np = np.zeros((B, 2, 2), dtype=np_dtype)

        for i in range(B):
            M_i = dyn.M(q_np[i])
            h_i = dyn.h(q_np[i], dq_np[i])
            qdd_np[i] = np.linalg.solve(M_i, (tau_np[i] - h_i))
            M_np[i] = M_i

        # reshape back
        qdd_np = qdd_np.reshape(q.shape[:-1] + (2,))
        M_np = M_np.reshape(q.shape[:-1] + (2, 2))

        M_t = torch.as_tensor(M_np, dtype=out_dtype, device=device)
        ctx.save_for_backward(M_t)
        ctx.batch_shape = q.shape[:-1]

        return torch.as_tensor(qdd_np, dtype=out_dtype, device=device)

    @staticmethod
    def backward(ctx, grad_qdd: torch.Tensor):
        (M_t,) = ctx.saved_tensors
        # grad_tau = solve(M^T, grad_qdd)
        grad_tau = torch.linalg.solve(M_t.transpose(-1, -2), grad_qdd.unsqueeze(-1)).squeeze(-1)
        return None, None, grad_tau, None, None


class FDLayerQddTauGrad(nn.Module):
    """
    Differentiable w.r.t tau ONLY.
    """

    def __init__(
        self,
        l1: float, l2: float,
        c1: float, c2: float,
        m1: float, m2: float,
        I1: float, I2: float,
        g: float = 9.81,
        np_dtype=np.float64,
    ):
        super().__init__()
        self.dyn = TwoLinkDynamics(
            l1_=l1, l2_=l2,
            c1_=c1, c2_=c2,
            m1_=m1, m2_=m2,
            I1_=I1, I2_=I2,
            g_=g,
            dtype=np_dtype,
        )
        self.np_dtype = np_dtype

    def forward(self, q: torch.Tensor, dq: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return _FDQddTauGradFn.apply(q, dq, tau, self.dyn, self.np_dtype)
