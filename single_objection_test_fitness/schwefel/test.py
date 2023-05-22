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

a = 418.9829 * m
val_max = 0.5  # 速度最值
x_var_max = 100  # 位置最值

start = cuda.Event()
end = cuda.Event()

x_cur = np.random.uniform(-x_var_max, x_var_max, (m, n)).astype(np.float32)
x_abs = np.zeros((m, n), np.float32)
x_sqrt = np.zeros((m, n), np.float32)
x_sin = np.zeros((m, n), np.float32)
y_one = np.zeros((n,), np.float32)
y_cur = np.zeros((n,), np.float32)  # 当前适应度值矩阵

x_cur_gpu = cuda.mem_alloc(x_cur.nbytes)
x_abs_gpu = cuda.mem_alloc(x_abs.nbytes)
x_sqrt_gpu = cuda.mem_alloc(x_sqrt.nbytes)
x_sin_gpu = cuda.mem_alloc(x_sin.nbytes)
y_one_gpu = cuda.mem_alloc(y_one.nbytes)
y_cur_gpu = cuda.mem_alloc(y_cur.nbytes)

cuda.memcpy_htod(x_cur_gpu, x_cur)
cuda.memcpy_htod(x_abs_gpu, x_abs)
cuda.memcpy_htod(x_sqrt_gpu, x_sqrt)
cuda.memcpy_htod(x_sin_gpu, x_sin)
cuda.memcpy_htod(y_one_gpu, y_one)
cuda.memcpy_htod(y_cur_gpu, y_cur)

fun_pso_str = """
__global__ void fun_abs(float *x, float * x_abs)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_abs[tx] = abs(x[tx]);
}
__global__ void fun_sqrt(float * x_abs, float * x_sqrt)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_sqrt[tx] = sqrt(x_abs[tx]);
}
__global__ void fun_sin(float * x_sqrt, float * x_sin)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_sin[tx] = sin(x_sqrt[tx]);
}
__global__ void fun_multi(float * x, float * x_sin)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_sin[tx] *= x[tx];
}
__device__ void fun_add_line(float *x, float *y)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y[tx] += x[tx];
}
__global__ void fun_sum(float y_temp[][%(n)s], float *y_cur)
{
    for(int line = 0; line < %(m)s; line++)
    {
        fun_add_line(y_temp[line], y_cur);
    }
}
__global__ void fun_y(float * y_one, float * y_cur)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y_cur[tx] = %(a)s - y_one[tx];
}
"""

fun_pso_str = fun_pso_str % {'n': n, 'm': m, 'a': a, }
mod_fun_obj = SourceModule(fun_pso_str)

fun_abs = mod_fun_obj.get_function('fun_abs')       # 矩阵元素绝对值
fun_sqrt = mod_fun_obj.get_function('fun_sqrt')     # 矩阵元素平方根
fun_sin = mod_fun_obj.get_function('fun_sin')       # 矩阵元素三角函数值
fun_multi = mod_fun_obj.get_function('fun_multi')   # 矩阵元素乘积
fun_sum = mod_fun_obj.get_function('fun_sum')       # 矩阵元素列求和
fun_y = mod_fun_obj.get_function('fun_y')       # 矩阵元素列求和

fun_abs(x_cur_gpu, x_abs_gpu, grid=grids, block=(n, 1, 1))
fun_sqrt(x_abs_gpu, x_sqrt_gpu, grid=grids, block=(n, 1, 1))
fun_sin(x_sqrt_gpu, x_sin_gpu, grid=grids, block=(n, 1, 1))
fun_multi(x_cur_gpu, x_sin_gpu, grid=grids, block=(n, 1, 1))
fun_sum(x_sin_gpu, y_one_gpu, grid=grids, block=(bn, 1, 1))
fun_y(y_one_gpu, y_cur_gpu, grid=grids, block=(bn, 1, 1))

print("初试设定的 x_cur: \n", x_cur)
cuda.memcpy_dtoh(x_sin, x_sin_gpu)
print("GPU 上计算的 x_sin：\n", x_sin)
cuda.memcpy_dtoh(x_cur, x_cur_gpu)
print("GPU 上计算的 x_cur：\n", x_cur)
cuda.memcpy_dtoh(y_one, y_one_gpu)
print("GPU 上计算的 y_one：\n", y_one, '\n')
cuda.memcpy_dtoh(y_cur, y_cur_gpu)
print("GPU 上计算的 y_cur：\n", y_cur)


def schwefel_fun(x_cur, y_cur):
    for j in range(n):
        sum = 0
        sum1 = 0
        for i in range(m):
            sum += x_cur[i][j] * np.sin(np.math.sqrt(abs(x_cur[i][j]))) * (-1)
            sum1 = a + sum
        y_cur[j] = sum1


schwefel_fun(x_cur, y_cur)
print("CPU 上计算的 y_cur: \n", y_cur)
