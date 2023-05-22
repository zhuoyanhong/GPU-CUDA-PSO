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


def sqrt_fun(x_sqrt):
    for j in range(n):
        for i in range(m):
            x_sqrt[i][j] = np.math.sqrt(i + 1)


sqrt_fun(x_sqrt)
x_sqrt = np.array(x_sqrt, np.float32)

x_cur_gpu = cuda.mem_alloc(x_cur.nbytes)
x_cur_sqr_gpu = cuda.mem_alloc(x_cur_sqr.nbytes)
x_sqr_sum_gpu = cuda.mem_alloc(x_sqr_sum.nbytes)
x_cur_div_gpu = cuda.mem_alloc(x_cur_div.nbytes)
x_div_multi_gpu = cuda.mem_alloc(x_div_multi.nbytes)
x_sqrt_gpu = cuda.mem_alloc(x_sqrt.nbytes)
y_cur_gpu = cuda.mem_alloc(y_cur.nbytes)

cuda.memcpy_htod(x_cur_gpu, x_cur)
cuda.memcpy_htod(x_cur_sqr_gpu, x_cur_sqr)
cuda.memcpy_htod(x_sqr_sum_gpu, x_sqr_sum)
cuda.memcpy_htod(x_cur_div_gpu, x_cur_div)
cuda.memcpy_htod(x_div_multi_gpu, x_div_multi)
cuda.memcpy_htod(x_sqrt_gpu, x_sqrt)
cuda.memcpy_htod(y_cur_gpu, y_cur)


fun_pso_str = """
__global__ void fun_sqr(float *x, float *x_sqr)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_sqr[tx] = x[tx] * x[tx];
}
__device__ void fun_add_line(float *x, float *y)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y[tx] += x[tx];
}
__global__ void fun_sum(float y_temp[][%(n)s], float *y_one)
{
    for(int line = 0; line < %(m)s; line++)
    {
        fun_add_line(y_temp[line], y_one);
    }
}
__global__ void fun_div(float * x_cur, float * x_sqrt, float * x_div)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_div[tx] = x_cur[tx] / x_sqrt[tx];
}
__device__ void fun_multi_line(float *x, float *y)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y[tx] *= x[tx];
}
__global__ void fun_multi(float y_temp[][%(n)s], float *y_one)
{
    for(int line = 0; line < %(m)s; line++)
    {
        fun_multi_line(y_temp[line], y_one);
    }
}
__global__ void fun_y(float * x_sqr_sum, float * x_div_multi, float * y_cur)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y_cur[tx] = %(a)s * x_sqr_sum[tx] - x_div_multi[tx] + 1;
}
"""

fun_pso_str = fun_pso_str % {'n': n, 'm': m, 'a': a, }
mod_fun_obj = SourceModule(fun_pso_str)

fun_sqr = mod_fun_obj.get_function('fun_sqr')  # 矩阵元素平方
fun_sum = mod_fun_obj.get_function('fun_sum')  # 矩阵列求和
fun_div = mod_fun_obj.get_function('fun_div')
fun_multi = mod_fun_obj.get_function('fun_multi')
fun_y = mod_fun_obj.get_function('fun_y')



print("最初的 x_cur: \n", x_cur)
fun_sqr(x_cur_gpu, x_cur_sqr_gpu, grid=grids, block=(n, 1, 1))
fun_sum(x_cur_sqr_gpu, x_sqr_sum_gpu, grid=grids, block=(bn, 1, 1))
fun_div(x_cur_gpu, x_sqrt_gpu, x_cur_div_gpu, grid=grids, block=(n, 1, 1))
fun_multi(x_cur_div_gpu, x_div_multi_gpu, grid=grids, block=(bn, 1, 1))
fun_y(x_sqr_sum_gpu, x_div_multi_gpu, y_cur_gpu, grid=grids, block=(bn, 1, 1))

cuda.memcpy_dtoh(x_cur_sqr, x_cur_sqr_gpu)
print("GPU 上计算的 x_cur_sqr：\n", x_cur_sqr)
cuda.memcpy_dtoh(x_sqr_sum, x_sqr_sum_gpu)
print("GPU 上计算的 x_sqr_sum：\n", x_sqr_sum)
cuda.memcpy_dtoh(x_sqr_sum, x_sqr_sum_gpu)
print("GPU 上计算的 x_sqr_sum：\n", x_sqr_sum)
cuda.memcpy_dtoh(x_cur_div, x_cur_div_gpu)
print("GPU 上计算的 x_cur_div：\n", x_cur_div)
cuda.memcpy_dtoh(x_div_multi, x_div_multi_gpu)
print("GPU 上计算的 x_div_multi：\n", x_div_multi)
cuda.memcpy_dtoh(y_cur, y_cur_gpu)
print("GPU 上计算的 y_cur：\n", y_cur)
