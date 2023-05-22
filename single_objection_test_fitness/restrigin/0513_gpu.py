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

a = 10
pi = np.pi
val_max = 0.5  # 速度最值
x_var_max = 5.12  # 位置最值

start = cuda.Event()
end = cuda.Event()

x_cur = np.random.uniform(-x_var_max, x_var_max, (m, n)).astype(np.float32)
x_cur_sqr = np.zeros((m, n), np.float32)
x_cur_cos = np.zeros((m, n), np.float32)
y_temp = np.zeros((m, n), np.float32)
y_cur = np.zeros((n,), np.float32)  # 当前适应度值矩阵

x_cur_gpu = cuda.mem_alloc(x_cur.nbytes)
x_cur_sqr_gpu = cuda.mem_alloc(x_cur_sqr.nbytes)
x_cur_cos_gpu = cuda.mem_alloc(x_cur_cos.nbytes)
y_temp_gpu = cuda.mem_alloc(y_temp.nbytes)
y_cur_gpu = cuda.mem_alloc(y_cur.nbytes)

cuda.memcpy_htod(x_cur_gpu, x_cur)
cuda.memcpy_htod(x_cur_sqr_gpu, x_cur_sqr)
cuda.memcpy_htod(x_cur_cos_gpu, x_cur_cos)
cuda.memcpy_htod(y_temp_gpu, y_temp)
cuda.memcpy_htod(y_cur_gpu, y_cur)

fun_pso_str = """
__global__ void fun_sqr(float *x, float *x_sqr)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_sqr[tx] = x[tx] * x[tx];
}
__global__ void fun_cos(float *x, float *x_cos)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_cos[tx] = %(a)s * cos(2 * %(pi)s * x[tx]);
}
__global__ void fun_sum(float * x_sqr, float * x_cos, float * y_temp)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y_temp[tx] = x_sqr[tx] - x_cos[tx] + %(a)s;
}
__device__ void fun_add_line(float *x, float *y)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y[tx] += x[tx];
}
__global__ void fun_add(float y_temp_1[][%(n)s], float *y_cur)
{
    for(int line = 0; line < %(m)s; line++)
    {
        fun_add_line(y_temp_1[line], y_cur);
    }
}
"""
fun_pso_str = fun_pso_str % {'n': n, 'm': m, 'a': a, 'pi': pi, }
mod_fun_obj = SourceModule(fun_pso_str)

fun_sqr = mod_fun_obj.get_function('fun_sqr')   # 矩阵元素平方
fun_cos = mod_fun_obj.get_function('fun_cos')   # 矩阵三角运算
fun_sum = mod_fun_obj.get_function('fun_sum')   # 矩阵求和运算
fun_add = mod_fun_obj.get_function('fun_add')   # 矩阵整列求和


if __name__ == '__main__':
    fun_sqr(x_cur_gpu, x_cur_sqr_gpu, grid=grids, block=(n, 1, 1))
    fun_cos(x_cur_gpu, x_cur_cos_gpu, grid=grids, block=(n, 1, 1))
    fun_sum(x_cur_sqr_gpu, x_cur_cos_gpu, y_temp_gpu, grid=grids, block=(n, 1, 1))
    fun_add(y_temp_gpu, y_cur_gpu, grid=grids, block=(bn, 1, 1))

    cuda.memcpy_dtoh(x_cur, x_cur_gpu)
    print("GPU 上计算的 x_cur :\n", x_cur)
    cuda.memcpy_dtoh(x_cur_sqr, x_cur_sqr_gpu)
    print("GPU 上计算的 x_cur_sqr :\n", x_cur_sqr)
    cuda.memcpy_dtoh(x_cur_cos, x_cur_cos_gpu)
    print("GPU 上计算的 x_cur_cos :\n", x_cur_cos)
    cuda.memcpy_dtoh(y_temp, y_temp_gpu)
    print("GPU 上计算的 y_temp :\n", y_temp)
    cuda.memcpy_dtoh(y_cur, y_cur_gpu)
    print("GPU 上计算的适应度值是：\n", y_cur)
    cuda.memcpy_dtoh(x_cur, x_cur_gpu)
    print("GPU 上计算的 x_cur :\n", x_cur)

