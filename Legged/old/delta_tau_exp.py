# rollout_delta_tau_analysis.py

import numpy as np
from legged_dynamics import LegDynamicsCached
from contact_pgs import resolve_toe_heel_pgs
from leg_env import LegEnv


# ---- 纯函数版 forward（和 LegEnv.step 一样，但不改 env） ----

def forward_with_contact(dyn, q, dq, tau,
                         dt=0.01, mu=0.8, beta=0.2, iters=25):
    """
    纯函数版 step: semi-implicit Euler + PGS。
    输入 q,dq,tau，输出 q_next, dq_next, contact_info。
    """
    q = np.asarray(q, float).reshape(6,)
    dq = np.asarray(dq, float).reshape(6,)
    tau = np.asarray(tau, float).reshape(6,)

    # free dynamics
    ddq_free = dyn.ddq(q, dq, tau)        # solve M ddq = tau - h
    dq_free = dq + dt * ddq_free

    # contact impulse correction
    dq_plus, contact = resolve_toe_heel_pgs(
        dyn, q, dq_free, dt,
        mu=mu, beta=beta, iters=iters
    )

    q_next = q + dt * dq_plus
    return q_next, dq_plus, contact


# ---- 一个简单的 PD policy（暂时在这个脚本里可选用） ----

def pd_stand_policy(obs, noise_std=2.0):
    """
    obs: [q(6), dq(6)]
    输出一个 torque，用来大致保持站立 + 小噪声。
    """
    q = obs[:6]
    dq = obs[6:]

    q_ref = np.array([0.0, 1.8, 0.0, 0.3, -0.6, 0.3], dtype=float)
    Kp = np.array([0.0, 0.0, 50.0, 60.0, 60.0, 40.0], dtype=float)
    Kd = np.array([0.0, 0.0,  5.0,  5.0,  5.0,  3.0], dtype=float)

    tau_pd = Kp * (q_ref - q) - Kd * dq
    noise = noise_std * np.random.randn(6)
    tau = tau_pd + noise
    return tau


# ---- 在单个 (q,dq,tau_base) 上 sample δτ 并对比 δdq ----

def analyze_delta_at_state(
    dyn,
    q, dq,
    tau_base,
    dt,
    mu,
    beta,
    iters_pgs,
    num_samples=16,
    delta_scale=5.0,
    print_prefix="",
):
    """
    在固定 state (q,dq) 和当前 torque tau_base 附近:
      - 求基准下一步 dq_next0 (用 tau_base)
      - sample 多个 δτ
      - 真正一步: dq_next1(tau_base + δτ)
        → δdq_true = dq_next1 - dq_next0
      - 线性近似: δdq_lin = dt * Minv @ δτ
      - 记录 norm / cos(angle)
    """
    q = np.asarray(q, float).reshape(6,)
    dq = np.asarray(dq, float).reshape(6,)
    tau_base = np.asarray(tau_base, float).reshape(6,)

    # 基准下一步速度
    _, dq_next0, contact0 = forward_with_contact(
        dyn, q, dq, tau_base,
        dt=dt, mu=mu, beta=beta, iters=iters_pgs
    )

    M = dyn.A(q, dq)
    Minv = np.linalg.inv(M)

    norm_true_list = []
    norm_lin_list = []
    cos_list = []

    for i in range(num_samples):
        delta_tau = delta_scale * np.random.randn(6)

        # 真一步（带 PGS）
        _, dq_next1, contact1 = forward_with_contact(
            dyn, q, dq, tau_base + delta_tau,
            dt=dt, mu=mu, beta=beta, iters=iters_pgs
        )
        delta_dq_true = dq_next1 - dq_next0

        # 线性近似
        delta_dq_lin = dt * (Minv @ delta_tau)

        n_true = np.linalg.norm(delta_dq_true)
        n_lin = np.linalg.norm(delta_dq_lin)
        denom = (n_true * n_lin + 1e-8)
        cos_th = float(delta_dq_true @ delta_dq_lin / denom)

        norm_true_list.append(n_true)
        norm_lin_list.append(n_lin)
        cos_list.append(cos_th)

        # 前几条打印细节
        if i < 4:
            print(f"{print_prefix}  sample {i}")
            print(f"{print_prefix}    delta_tau    = {np.round(delta_tau, 3)}")
            print(f"{print_prefix}    δdq_true     = {np.round(delta_dq_true, 5)}")
            print(f"{print_prefix}    δdq_lin      = {np.round(delta_dq_lin, 5)}")
            print(f"{print_prefix}    |true|={n_true:.5f}, |lin|={n_lin:.5f}, cos={cos_th:.4f}")
            # 如要看接触模式，可以解开：
            # print(f"{print_prefix}    contact1: {contact1}")

    summary = {
        "mean_true": float(np.mean(norm_true_list)),
        "mean_lin": float(np.mean(norm_lin_list)),
        "mean_cos": float(np.mean(cos_list)),
        "min_cos": float(np.min(cos_list)),
        "max_cos": float(np.max(cos_list)),
    }

    print(f"{print_prefix}  summary: "
          f"mean|δdq_true|={summary['mean_true']:.5f}, "
          f"mean|δdq_lin|={summary['mean_lin']:.5f}, "
          f"mean cos={summary['mean_cos']:.4f}, "
          f"min/max cos=({summary['min_cos']:.4f}, {summary['max_cos']:.4f})")

    return summary


# ---- rollout + 一边走一边在接触附近做 δτ 分析 ----

def rollout_with_delta_analysis(
    dyn,
    dt=0.01,
    mu=0.8,
    beta=0.2,
    iters_pgs=25,
    horizon=300,
    delta_scale=5.0,
    num_delta_samples=16,
    noise_std=2.0,
    y_tol=0.05,          # y < y_tol 认为“接近地面”
):
    env = LegEnv(dyn, dt=dt, mu=mu, beta=beta, iters=iters_pgs)
    obs = env.reset(randomize=True)

    started_analysis = False

    for step in range(horizon):
        q = env.q.copy()
        dq = env.dq.copy()

        # 你可以先用 0 torque 看自由落体 + 接触
        tau_base = np.zeros(6, dtype=float)
        # 或者用 PD 站立（解开下面这行，注释上一行）
        # tau_base = pd_stand_policy(obs, noise_std=noise_std)

        # 看脚 y 高度
        toe_y = dyn.pToe(q)[1]
        heel_y = dyn.pHeel(q)[1]

        # 还在天上就纯 rollout，不分析
        if (toe_y > y_tol) and (heel_y > y_tol) and not started_analysis:
            obs_next, done, info = env.step(tau_base)
            obs = obs_next
            if done:
                print(f"Terminated before contact at step {step}.")
                break
            continue
        else:
            if not started_analysis:
                print(f"\n=== start analyzing at step {step}, "
                      f"toe_y={toe_y:.3f}, heel_y={heel_y:.3f} ===")
                started_analysis = True

        print(f"\n=== step {step} ===")
        print("q        =", np.round(q, 3))
        print("dq       =", np.round(dq, 3))
        print("tau_base =", np.round(tau_base, 3))
        print(f"toe_y={toe_y:.4f}, heel_y={heel_y:.4f}")

        # 在当前 (q,dq,tau_base) 附近 sample δτ，并对比 δdq
        analyze_delta_at_state(
            dyn,
            q, dq,
            tau_base,
            dt=dt,
            mu=mu,
            beta=beta,
            iters_pgs=iters_pgs,
            num_samples=num_delta_samples,
            delta_scale=delta_scale,
            print_prefix=f"[step {step}]",
        )

        # 正式环境 step
        obs_next, done, info = env.step(tau_base)
        obs = obs_next
        if done:
            print(f"Episode terminated at step {step} (safety).")
            break


if __name__ == "__main__":
    # 和你 sympy dynamics 一致的参数
    params = dict(
        M=3.0, m1=1.5, m2=1.0, m3=2.0,
        I=0.01, I1=0.005, I2=0.005, I3=0.009,
        l0=0.3, l1=0.6, l2=0.6, l3=0.5,
        c1=0.3, c2=0.2, c3=0.4,
        g=9.81
    )
    dyn = LegDynamicsCached(params)

    rollout_with_delta_analysis(
        dyn,
        dt=0.01,
        mu=0.8,
        beta=0.2,
        iters_pgs=25,
        horizon=300,         # 可以加长一点，看多点接触阶段
        delta_scale=5.0,     # 可以试 0.5 / 1.0 / 10.0 比较线性程度
        num_delta_samples=16,
        noise_std=2.0,
        y_tol=0.05,
    )
