import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit


np.random.seed(42)

m = 2 ** 6   # 维度
n = 2 ** 9  # 粒子数  n = bm * bn
bm = 2 ** 6
bn = 2 ** 3
grids = (2 ** 6, 1, 1)

iter_num = 2000
rand_trans_steps = 2 * (10 ** 3)

w = 0.4
c1, c2 = np.float32(2.), np.float32(2.)
r1, r2 = np.random.random((2, rand_trans_steps, 1, n)).astype(np.float32)
offset_rand = r1.nbytes / rand_trans_steps * m

pi = np.pi
val_max = 0.5  # 速度最值
x_var_max = 2 ** pi  # 位置最值

start = cuda.Event()
end = cuda.Event()

x_cur = np.random.uniform(-x_var_max, x_var_max, (m, n)).astype(np.float32)
print("初始产生的 x_cur: \n", x_cur)
y_cur = np.zeros((n,), np.float32)  # 当前适应度值矩阵


def fitness_func(x_cur_0, y_cur_0):
    for j in range(n):
        sum = 0
        b = 0
        c = 1
        for i in range(m):
            sum += (-1) * (x_cur_0[i][j] - pi) ** 2
            b = np. exp(sum)
            c *= np.cos(x_cur_0[i][j]) ** 2
        y_cur_0[j] = ((-1) ** m) * b * c


fitness_func(x_cur, y_cur)
print("CPU 上计算的 y_cur: \n", y_cur)


