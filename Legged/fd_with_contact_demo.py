import numpy as np
from legged_dynamics import LegDynamicsCached
import time
import cv2
from sim import physics_step

def loop_show(dyn, q, dq, dt=0.01, mu=0.8, beta=0.2, iters=25, W=480, H=480, render_num=2):
    counter = 0
    for _ in range(1000):
        start_time = time.time()
        q, dq, contact = physics_step(dyn, q, dq, dt, mu=mu, beta=beta, iters=iters)
        print(q)
        if counter < render_num: 
            counter += 1
        else:
            old = time.time()
            rgb = dyn.render_rgb(q)
            cv2.imshow("sim", rgb[:, :, ::-1])          # BGR
            if (cv2.waitKey(1) & 0xFF) == 27: break     # ESC
            print("render:", time.time() - old)
            counter = 0
        if time.time() - start_time < dt: 
            time.sleep(dt - (time.time() - start_time))
    cv2.destroyAllWindows()

if __name__ == "__main__":  
    q0  = np.array([0.0, 3.0, 0.0, 0.3, -0.6, 0.2])
    dq0 = np.zeros(6)
    params = dict(
    M=3.0, m1=1.5, m2=1.0, m3=2.0,
    I=0.01, I1=0.005, I2=0.005, I3=0.009,
    l0=0.3, l1=0.6, l2=0.6, l3=0.5,
    c1=0.3, c2=0.2, c3=0.4,
    g=9.81
    )
    dyn = LegDynamicsCached(params)
    loop_show(dyn, q0, dq0)
