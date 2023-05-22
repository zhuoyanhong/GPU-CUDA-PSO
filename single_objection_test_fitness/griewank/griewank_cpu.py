import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit


np.random.seed(42)

m = 2 ** 5   # 维度
n = 2 ** 9  # 粒子数  n = bm * bn


iter_num = 1
rand_trans_steps = 2 * (10 ** 3)

w = 0.4
c1, c2 = np.float32(2.), np.float32(2.)
r1, r2 = np.random.random((2, rand_trans_steps, 1, n)).astype(np.float32)
offset_rand = r1.nbytes / rand_trans_steps * m

a = 1/4000
val_max = 0.5  # 速度最值
x_var_max = 600  # 位置最值

start = cuda.Event()
end = cuda.Event()

x_cur = np.random.uniform(-x_var_max, x_var_max, (m, n)).astype(np.float32)
x_cur_sqr = np.zeros((m, n), np.float32)
x_sqr_sum = np.zeros((n, ), np.float32)
x_cur_div = np.zeros((m, n), np.float32)
x_div_multi = np.zeros((n, ), np.float32)
x_sqrt = np.zeros((m, n), np.float32)
y_cur = np.zeros((n,), np.float32)  # 当前适应度值矩阵


def x_sqr_sum_fun(x_cur, x_cur_sqr, x_sqr_sum):
    for j in range(n):
        sum = 0
        for i in range(m):
            x_cur_sqr[i][j] = x_cur[i][j] ** 2
            sum += x_cur_sqr[i][j]
        x_sqr_sum[j] = sum


x_sqr_sum_fun(x_cur, x_cur_sqr, x_sqr_sum)
print("CPU 上计算的 x_sqr_sum: \n", x_sqr_sum)


def sqrt_fun(x_sqrt):
    for j in range(n):
        for i in range(m):
            x_sqrt[i][j] = np.math.sqrt(i + 1)


sqrt_fun(x_sqrt)
print("CPU 上计算的 x_sqrt: \n", x_sqrt)


def div_fun(x_cur_0, x_sqrt_0, x_div_0):
    for j in range(n):
        for i in range(m):
            x_div_0[i][j] = x_cur_0[i][j] / x_sqrt_0[i][j]


div_fun(x_cur, x_sqrt, x_cur_div)
print('CPU 上计算的 x_cur_div:\n', x_cur_div)


def multi_sum_fun(x_div, x_div_multi):
    for j in range(n):
        sum = 1
        for i in range(m):
            sum *= x_div[i][j]
        x_div_multi[j] = sum


multi_sum_fun(x_cur_div, x_div_multi)
print('CPU 上计算的 x_div_multi:\n', x_div_multi)


def fun_y(x_sqr_sum, x_div_multi, y_cur):
    for j in range(n):
        sum = 0
        sum += a * x_sqr_sum[j] - x_div_multi[j] + 1
        y_cur[j] = sum


fun_y(x_sqr_sum, x_div_multi, y_cur)
print("CPU 上计算的 y_cur：\n", y_cur)


# def fitness_fun(x_cur_0, y_cur_0):
#     for j in range(n):
#         sum0 = 0
#         sum1 = 1
#         for i in range(m):
#             sum0 += a * x_cur_0[i][j] ** 2
#             b = np.math.sqrt(i + 1)
#             c = x_cur_0[i][j] / b
#             sum1 *= c
#         y_cur_0[j] = sum0 - sum1 + 1
#
#
# fitness_fun(x_cur, y_cur)
# print("CPU 上计算的 y_cur：\n", y_cur)
