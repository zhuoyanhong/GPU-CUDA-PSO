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

a = 10 ** 6
pow = np.math.pow
val_max = 0.5  # 速度最值
x_var_max = 100  # 位置最值

start = cuda.Event()
end = cuda.Event()


x_cur = np.random.uniform(-x_var_max, x_var_max, (m, n)).astype(np.float32)
x_cur_sqr = np.zeros((m, n), np.float32)
x_coef = a * np.ones((m, n), np.float32)
x_expo = np.zeros((m, n), np.float32)
y_one = np.zeros((m, n), np.float32)
y_cur = np.zeros((n,), np.float32)  # 当前适应度值矩阵

print("CPU 上计算所得的 x_cur:\n", x_cur)


# def x_sqr_fun(x_cur, x_sqr):
#     for j in range(n):
#         for i in range(m):
#             x_sqr[i][j] = x_cur[i][j] ** 2
#
#
# x_sqr_fun(x_cur, x_cur_sqr)
# print("CPU 上计算所得的 x_cur_sqr:\n", x_cur_sqr)
#
#
# def x_expo_fun(x_expo):
#     for i in range(m):
#         for j in range(n):
#             x_expo[i][j] = i/(m - 1)
#
#
# x_expo_fun(x_expo)
# print("CPU 上计算所得的 x_expo:\n", x_expo)
#
#
# def x_coef_fun(x_coef, x_expo):
#     for j in range(n):
#         for i in range(m):
#             x_coef[i][j] = np.math.pow(x_coef[i][j], x_expo[i][j])
#
#
# x_coef_fun(x_coef, x_expo)
# print("CPU 上计算所得的 x_coef:\n", x_coef)
#
#
# def sqr_coef_multi(x_coef, x_sqr, y_one):
#     for j in range(n):
#         for i in range(m):
#             y_one[i][j] = x_coef[i][j] * x_sqr[i][j]
#
#
# sqr_coef_multi(x_coef, x_cur_sqr, y_one)
# print("CPU 上计算所得的 y_one:\n", y_one)
#
#
# def y_fun(y_one, y_cur):
#     for j in range(n):
#         sum = 0
#         for i in range(m):
#             sum += y_one[i][j]
#         y_cur[j] = sum
#
#
# y_fun(y_one, y_cur)
# print("CPU 上计算所得的 y_cur:\n", y_cur)


def fitness_fun(x_cur_0, x_expo_0, x_coef_0, y_cur_0):
    x_sqr_0 = x_cur_0 ** 2
    for i in range(m):
        for j in range(n):
            x_expo_0[i][j] = i/(m - 1)
            x_coef_0[i][j] = np.math.pow(x_coef_0[i][j], x_expo_0[i][j])
    y_one_0 = x_coef_0 * x_sqr_0
    for j in range(n):
        sum = 0
        for i in range(m):
            sum += y_one_0[i][j]
        y_cur_0[j] = sum


fitness_fun(x_cur, x_expo, x_coef, y_cur)
print("CPU 上计算所得的 y_cur:\n", y_cur)
