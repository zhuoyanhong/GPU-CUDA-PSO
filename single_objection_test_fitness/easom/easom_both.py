import numpy as np
import gc
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
import pycuda.autoinit

np.random.seed(42)

iter_num = 2000
rand_trans_steps = 2 * (10 ** 3)

m = 2 ** 6   # 维度
n = 2 ** 9  # 粒子数  n = bm * bn
bm = 2 ** 6
bn = 2 ** 3
grids = (2 ** 6, 1, 1)

pi = np.pi
x_var_max = 2 * pi
# print(x_var_max)
x_cur = np.random.uniform(-x_var_max, x_var_max, (m, n)).astype(np.float32)
print("初始产生的 x_cur: \n", x_cur)
x_cur_sqr = np.zeros((m, n), np.float32)
x_sqr_sum = np.zeros((1, n), np.float32)
x_sum_exp = np.zeros((1, n), np.float32)
x_cos = np.zeros((m, n), np.float32)
x_cos_sqr = np.zeros((m, n), np.float32)
x_cos_multi = np.ones((1, n), np.float32)
y_cur = np.zeros((n, ), np.float32)


x_cur_gpu = cuda.mem_alloc(x_cur.nbytes)
x_cur_sqr_gpu = cuda.mem_alloc(x_cur_sqr.nbytes)
x_sqr_sum_gpu = cuda.mem_alloc(x_sqr_sum.nbytes)
x_sum_exp_gpu = cuda.mem_alloc(x_sum_exp.nbytes)
x_cos_gpu = cuda.mem_alloc(x_cos.nbytes)
x_cos_sqr_gpu = cuda.mem_alloc(x_cos_sqr.nbytes)
x_cos_multi_gpu = cuda.mem_alloc(x_cos_multi.nbytes)
y_cur_gpu = cuda.mem_alloc(y_cur.nbytes)

cuda.memcpy_htod(x_cur_gpu, x_cur)
cuda.memcpy_htod(x_cur_sqr_gpu, x_cur_sqr)
cuda.memcpy_htod(x_sqr_sum_gpu, x_sqr_sum)
cuda.memcpy_htod(x_sum_exp_gpu, x_sum_exp)
cuda.memcpy_htod(x_cos_gpu, x_cos)
cuda.memcpy_htod(x_cos_sqr_gpu, x_cos_sqr)
cuda.memcpy_htod(x_cos_multi_gpu, x_cos_multi)
cuda.memcpy_htod(y_cur_gpu, y_cur)

fun_pso_str = """
__global__ void fun_sqr(float *x, float *x_sqr)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_sqr[tx] = (x[tx] - %(pi)s) * (x[tx] - %(pi)s);
}
__device__ void fun_add_line(float *x, float *y)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y[tx] += x[tx];
}
__global__ void fun_add(float x_sqr[][%(n)s], float *x_sum)
{
    for(int line = 0; line < %(m)s-1; line++)
    {
        fun_add_line(x_sqr[line], x_sum);
    }
}
__global__ void fun_exp(float *x_sum, float * x_exp)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_exp[tx] = exp(x_sum[tx] * (-1));
}
__global__ void fun_cos(float * x_cur, float * x_cos)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_cos[tx] = cos(x_cur[tx]);
}
__global__ void fun_cos_sqr(float *x, float *x_sqr)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_sqr[tx] = (x[tx]) * (x[tx]);
}
__device__ void fun_multi_line(float *x, float *y)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y[tx] *= x[tx];
}
__global__ void fun_multi(float x_cos[][%(n)s], float *x_multi)
{
    for(int line = 0; line < %(m)s-1; line++)
    {
        fun_multi_line(x_cos[line], x_multi);
    }
}
__global__ void fun_y(float * x_sum, float * x_multi, float * y_cur)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y_cur[tx] = x_sum[tx] * x_multi[tx];
}
"""

fun_pso_str = fun_pso_str % {'n': n, 'm': m, 'pi': pi, }
mod_fun_obj = SourceModule(fun_pso_str)

fun_sqr = mod_fun_obj.get_function('fun_sqr')   # 矩阵元素平方
fun_add = mod_fun_obj.get_function('fun_add')   # 矩阵列求和
fun_exp = mod_fun_obj.get_function('fun_exp')   # 矩阵指数

fun_cos = mod_fun_obj.get_function('fun_cos')   # 矩阵元素cos(x)
fun_cos_sqr = mod_fun_obj.get_function('fun_cos_sqr')
fun_multi = mod_fun_obj.get_function('fun_multi')   # 矩阵列求积

fun_y = mod_fun_obj.get_function('fun_y')   # 最终适应度值

fun_sqr(x_cur_gpu, x_cur_sqr_gpu, grid=grids, block=(n, 1, 1))
fun_add(x_cur_sqr_gpu, x_sqr_sum_gpu, grid=grids, block=(bn, 1, 1))
fun_exp(x_sqr_sum_gpu, x_sum_exp_gpu, grid=grids, block=(bn, 1, 1))

fun_cos(x_cur_gpu, x_cos_gpu, grid=grids, block=(n, 1, 1))
fun_cos_sqr(x_cos_gpu, x_cos_sqr_gpu, grid=grids, block=(n, 1, 1))
fun_multi(x_cos_sqr_gpu, x_cos_multi_gpu, grid=grids, block=(bn, 1, 1))

fun_y(x_sum_exp_gpu, x_cos_multi_gpu, y_cur_gpu, grid=grids, block=(bn, 1, 1))

cuda.memcpy_dtoh(x_cur_sqr, x_cur_sqr_gpu)
cuda.memcpy_dtoh(x_sqr_sum, x_sqr_sum_gpu)
cuda.memcpy_dtoh(x_sum_exp, x_sum_exp_gpu)
# print("GPU 上计算的 x_cur_sqr: \n", x_cur_sqr)
# print("GPU 上计算的 x_sqr_sum: \n", x_sqr_sum)
# print("GPU 上计算的 x_sum_exp: \n", x_sum_exp)

cuda.memcpy_dtoh(x_cos, x_cos_gpu)
cuda.memcpy_dtoh(x_cos_sqr, x_cos_sqr_gpu)
cuda.memcpy_dtoh(x_cos_multi, x_cos_multi_gpu)
# print("GPU 上计算的 x_cos: \n", x_cos)
# print("GPU 上计算的 x_cos_sqr: \n", x_cos_sqr)
# print("GPU 上计算的 x_cos_multi: \n", x_cos_multi)

cuda.memcpy_dtoh(y_cur, y_cur_gpu)
print("GPU 上计算的 y_cur: \n", y_cur)


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
