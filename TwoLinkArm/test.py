from twolink_dynamics import TwoLinkDynamics
import numpy as np

dyn = TwoLinkDynamics(
    l1_=1.0, l2_=1.0,
    c1_=0.5, c2_=0.5,
    m1_=1.0, m2_=1.0,
    I1_=0.1, I2_=0.1,
    g_=9.81
)

q  = np.array([0.3, -0.2])
dq = np.array([0.1,  0.0])
tau= np.array([0.0,  0.0])

print("M:\n", dyn.M(q))
print("h:", dyn.h(q,dq))
print("ddq:", dyn.ddq(q,dq,tau))
print("pL1:", dyn.link1_end(q))
print("pEE:", dyn.ee(q))
print("JL1:\n", dyn.J_link1_end(q))
print("JEE:\n", dyn.J_ee(q))
print("vEE:", dyn.J_ee(q) @ dq)   # linear end-effector velocity
