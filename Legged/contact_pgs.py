import numpy as np

def resolve_toe_heel_pgs(dyn, q, dq, dt, mu=0.1, beta=0.2,
                         phi_slop=1e-4, iters=25, eps=1e-8):
    """
    Flat ground y=0. Two contacts: toe & heel.
    Returns dq_plus and impulses [(lt, ln) for each active contact] with names.
    """
    q = np.asarray(q, float).reshape(-1,)
    dq = np.asarray(dq, float).reshape(-1,)
    n = dq.size

    M = dyn.A(q, dq)
    Minv = np.linalg.inv(M)

    # contact data: (name, y(q), J(2xn))
    contacts = [
        ("toe",  dyn.pToe(q, dq)[1],  dyn.JToe(q, dq)),
        ("heel", dyn.pHeel(q, dq)[1], dyn.JHeel(q, dq)),
    ]

    tdir = np.array([1., 0.])
    ndir = np.array([0., 1.])

    # build active pairs: (name, Jt(1xn), Jn(1xn), bt, bn)
    pairs = []
    for name, y, J2 in contacts:
        Jt = (tdir @ J2).reshape(n,)
        Jn = (ndir @ J2).reshape(n,)
        vn = float(Jn @ dq)

        # activation + bn rule
        if y < 0.0:
            bn = max(0.0, beta * (-y) / max(dt, 1e-12))
        elif (y <= phi_slop) and (vn < 0.0):
            bn = 0.0
        else:
            continue  # inactive => no impulse

        pairs.append((name, Jt, Jn, 0.0, bn))  # bt=0 => stick preference

    if not pairs:
        return dq.copy(), {}

    K = len(pairs)
    lt = np.zeros(K); ln = np.zeros(K)
    is_slipping = [True]*K

    # diagonal effective masses
    Wtt = np.array([Jt @ (Minv @ Jt) for _, Jt, _, _, _ in pairs], float)
    Wnn = np.array([Jn @ (Minv @ Jn) for _, _, Jn, _, _ in pairs], float)

    v = dq.copy()
    for _ in range(iters):
        for k, (name, Jt, Jn, bt, bn) in enumerate(pairs):
            # ---- normal ----
            dln = (bn - (Jn @ v)) / (Wnn[k] + eps)
            ln_old = ln[k]
            ln[k] = max(0.0, ln_old + dln)
            v += (Minv @ Jn) * (ln[k] - ln_old)

            # ---- tangential ----
            dlt = (bt - (Jt @ v)) / (Wtt[k] + eps)
            lt_old = lt[k]
            bound = mu * ln[k]
            lt_free = lt_old + dlt
            lt[k] = 0.0 if bound <= 0.0 else float(np.clip(lt_free, -bound, bound))
            is_slipping[k] = (abs(lt[k]) >= bound - 1e-8)
            v += (Minv @ Jt) * (lt[k] - lt_old)

    dq_plus = v
    info = {pairs[k][0]: {"lt": float(lt[k]), "Jt": pairs[k][1], "ln": float(ln[k]), "Jn": pairs[k][2], "is_slipping": bool(is_slipping[k]), "mu": mu} for k in range(K)}
    return dq_plus, info
