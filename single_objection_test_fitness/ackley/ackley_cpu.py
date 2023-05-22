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

iter_num = 1
rand_trans_steps = 2 * (10 ** 3)

w = 0.7
c1, c2 = np.float32(2.), np.float32(2.)
r1, r2 = np.random.random((2, rand_trans_steps, 1, n)).astype(np.float32)
offset_rand = r1.nbytes / rand_trans_steps * m

pi = np.pi
e = np.exp(1)
val_max = 0.8  # 速度最值
x_var_max = 30  # 位置最值

start = cuda.Event()
end = cuda.Event()

x_cur = np.random.uniform(-x_var_max, x_var_max, (m, n)).astype(np.float32)
x_cur_sqr = np.zeros((m, n), np.float32)
x_cos = np.zeros((m, n), np.float32)
x_cos_div = np.zeros((m, n), np.float32)
x_cos_sum = np.zeros((n, ), np.float32)
x_sqr_sum = np.zeros((n, ), np.float32)
x_sqr_sum_div = np.zeros((n, ), np.float32)
x_sqr_sum_div_sqrt = np.zeros((n, ), np.float32)
y_cur = np.zeros((n,), np.float32)  # 当前适应度值矩阵

x_cur_gpu = cuda.mem_alloc(x_cur.nbytes)
x_cur_sqr_gpu = cuda.mem_alloc(x_cur_sqr.nbytes)
x_cos_gpu = cuda.mem_alloc(x_cos.nbytes)
x_cos_div_gpu = cuda.mem_alloc(x_cos_div.nbytes)
x_cos_sum_gpu = cuda.mem_alloc(x_cos_sum.nbytes)
x_sqr_sum_gpu = cuda.mem_alloc(x_sqr_sum.nbytes)
x_sqr_sum_div_gpu = cuda.mem_alloc(x_sqr_sum_div.nbytes)
x_sqr_sum_div_sqrt_gpu = cuda.mem_alloc(x_sqr_sum_div_sqrt.nbytes)
y_cur_gpu = cuda.mem_alloc(y_cur.nbytes)

cuda.memcpy_htod(x_cur_gpu, x_cur)
cuda.memcpy_htod(x_cur_sqr_gpu, x_cur_sqr)
cuda.memcpy_htod(x_cos_gpu, x_cos)
cuda.memcpy_htod(x_cos_div_gpu, x_cos_div)
cuda.memcpy_htod(x_cos_sum_gpu, x_cos_sum)
cuda.memcpy_htod(x_sqr_sum_gpu, x_sqr_sum)
cuda.memcpy_htod(x_sqr_sum_div_gpu, x_sqr_sum_div)
cuda.memcpy_htod(x_sqr_sum_div_sqrt_gpu, x_sqr_sum_div_sqrt)
cuda.memcpy_htod(y_cur_gpu, y_cur)


def ackely_fun_two(x_cur, x_cos, x_cos_div, x_cos_sum):
    for j in range(n):
        sum0 = 0
        for i in range(m):
            x_cos[i][j] = np.cos(2 * pi * x_cur[i][j])
            x_cos_div[i][j] = x_cos[i][j] / m
            sum0 += x_cos_div[i][j]
        x_cos_sum[j] = sum0


def ackely_fun_one(x_cur, x_cur_sqr, x_sqr_sum, x_sqr_sum_div, x_sqr_sum_div_sqrt):
    for j in range(n):
        sum1 = 0
        for i in range(m):
            x_cur_sqr[i][j] = x_cur[i][j] ** 2
            sum1 += x_cur_sqr[i][j]
        x_sqr_sum[j] = sum1
        x_sqr_sum_div[j] = x_sqr_sum[j] / m
        x_sqr_sum_div_sqrt[j] = np.math.sqrt(x_sqr_sum_div[j])


def ackely_fun(x_sqr_sum_div_sqrt, x_cos_sum, y_cur):
    for j in range(n):
        y_cur[j] = 20 + e - 20 * np.exp(-0.2 * x_sqr_sum_div_sqrt[j]) - np.exp(x_cos_sum[j])


ackely_fun_two(x_cur, x_cos, x_cos_div, x_cos_sum)
print("CPU 上计算的 x_cos: \n", x_cos)
print("CPU 上计算的 x_cos_div: \n", x_cos_div)
print("CPU 上计算的 x_cos_sum: \n", x_cos_sum)

ackely_fun_one(x_cur, x_cur_sqr, x_sqr_sum, x_sqr_sum_div, x_sqr_sum_div_sqrt)
print("CPU 上计算的 x_cur_sqr: \n", x_cur_sqr)
print("CPU 上计算的 x_sqr_sum: \n", x_sqr_sum)
print("CPU 上计算的 x_sqr_sum_div: \n", x_sqr_sum_div)
print("CPU 上计算的 x_sqr_sum_div_sqrt: \n", x_sqr_sum_div_sqrt)

ackely_fun(x_sqr_sum_div_sqrt, x_cos_sum, y_cur)
print("CPU 上计算的 y_cur: \n", y_cur)