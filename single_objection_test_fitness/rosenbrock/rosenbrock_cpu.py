import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit


np.random.seed(42)

m = 2 ** 8   # 维度
n = 2 ** 10  # 粒子数  n = bm * bn
bm = 2 ** 8
bn = 2 ** 2
grids = (2 ** 8, 1, 1)

iter_num = 2000
rand_trans_steps = 2 * (10 ** 3)

w = 0.4
c1, c2 = np.float32(2.), np.float32(2.)
r1, r2 = np.random.random((2, rand_trans_steps, 1, n)).astype(np.float32)
offset_rand = r1.nbytes / rand_trans_steps * m

val_max = 0.5  # 速度最值
x_var_max = 5  # 位置最值

start = cuda.Event()
end = cuda.Event()

x_cur = np.random.uniform(-x_var_max, x_var_max, (m, n)).astype(np.float32)
y_cur = np.zeros((n,), np.float32)  # 当前适应度值矩阵


def rosenbrock_fun(x_cur, y_cur):
    for j in range(n):
        sum = 0
        for i in range(m-1):
            sum += 100 * (x_cur[i][j]**2 - x_cur[i+1][j]) ** 2 + (x_cur[i][j] - 1) ** 2
        y_cur[j] = sum


if __name__ == '__main__':
    rosenbrock_fun(x_cur, y_cur)
    print("CPU 上计算的适应度值是：", y_cur)