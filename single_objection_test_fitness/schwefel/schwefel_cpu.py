import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit


np.random.seed(42)

m = 2 ** 6   # 维度
n = 2 ** 8  # 粒子数  n = bm * bn
bm = 2 ** 6
bn = 2 ** 2
grids = (2 ** 6, 1, 1)

iter_num = 2000
rand_trans_steps = 2 * (10 ** 3)

w = 0.4
c1, c2 = np.float32(2.), np.float32(2.)
r1, r2 = np.random.random((2, rand_trans_steps, 1, n)).astype(np.float32)
offset_rand = r1.nbytes / rand_trans_steps * m

a = 418.9829 * m
val_max = 0.5  # 速度最值
x_var_max = 500  # 位置最值

start = cuda.Event()
end = cuda.Event()

x_cur = np.random.uniform(-x_var_max, x_var_max, (m, n)).astype(np.float32)
x_cur_sqr = np.zeros((m, n), np.float32)
x_one = np.zeros((m, n), np.float32)
x_two = np.zeros((m, n), np.float32)
y_cur = np.zeros((n,), np.float32)  # 当前适应度值矩阵

x_cur_gpu = cuda.mem_alloc(x_cur.nbytes)
x_cur_sqr_gpu = cuda.mem_alloc(x_cur_sqr.nbytes)
x_one_gpu = cuda.mem_alloc(x_one.nbytes)
x_two_gpu = cuda.mem_alloc(x_two.nbytes)
y_cur_gpu = cuda.mem_alloc(y_cur.nbytes)

cuda.memcpy_htod(x_cur_gpu, x_cur)
cuda.memcpy_htod(x_cur_sqr_gpu, x_cur_sqr)
cuda.memcpy_htod(x_one_gpu, x_one)
cuda.memcpy_htod(x_two_gpu, x_two)
cuda.memcpy_htod(y_cur_gpu, y_cur)


def schwefel_fun(x_cur, y_cur):
    for j in range(n):
        sum = 0
        sum1 = 0
        for i in range(m):
            sum += x_cur[i][j] * np.sin(np.math.sqrt(abs(x_cur[i][j]))) * (-1)
            sum1 = a + sum
        y_cur[j] = sum1


print("初始设定的 x_cur: \n", x_cur)
schwefel_fun(x_cur, y_cur)
print("CPU 上计算的 y_cur: \n", y_cur)
