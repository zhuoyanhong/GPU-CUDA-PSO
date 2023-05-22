import numpy as np
import gc
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
x_var_max = 10  # 位置最值

start = cuda.Event()
end = cuda.Event()

x_cur = np.random.uniform(-x_var_max, x_var_max, (m, n)).astype(np.float32)
x_cur_gpu = cuda.mem_alloc(x_cur.nbytes)
cuda.memcpy_htod(x_cur_gpu, x_cur)

x_cur_sqr = np.zeros((m, n), np.float32)
x_cur_sqr_gpu = cuda.mem_alloc(x_cur_sqr.nbytes)
cuda.memcpy_htod(x_cur_sqr_gpu, x_cur_sqr)

y_cur = np.zeros((n,), np.float32)  # 当前适应度值矩阵
y_cur_gpu = cuda.mem_alloc(y_cur.nbytes)
cuda.memcpy_htod(y_cur_gpu, y_cur)


fun_pso_str = """
__global__ void fun_sqr(float *x, float *y_temp)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y_temp[tx] = x[tx] * x[tx];
}
__device__ void fun_add_line(float *x, float *y)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y[tx] += x[tx];
}
__global__ void fun_add(float y_temp[][%(n)s], float *y_cur)
{
    for(int line = 0; line < %(m)s; line++)
    {
        fun_add_line(y_temp[line], y_cur);
    }
}
"""
fun_pso_str = fun_pso_str % {'n': n, 'm': m, }
mod_fun_obj = SourceModule(fun_pso_str)

fun_sqr = mod_fun_obj.get_function('fun_sqr')  # 矩阵元素平方
fun_add = mod_fun_obj.get_function('fun_add')  # 矩阵列求和


def fitness_func(H, y_cur):
    for j in range(n):
        sum = 0
        for i in range(m):
            sum += H[i][j]**2
        y_cur[j] = sum



print(x_cur)
# 计算适应度值  y_cur
fun_sqr(x_cur_gpu, x_cur_sqr_gpu, grid=grids, block=(n, 1, 1))
fun_add(x_cur_sqr_gpu, y_cur_gpu, grid=grids, block=(bn, 1, 1))
cuda.memcpy_dtoh(y_cur, y_cur_gpu)
print("GPU 上计算的适应度值是：", y_cur)

fitness_func(x_cur, y_cur)
print("CPU 上计算的适应度值是：", y_cur)