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
x_cur_sqr = np.ones((m, n), np.float32)
x_coef = a * np.ones((m, n), np.float32)
y_one = np.zeros((m, n), np.float32)
y_cur = np.zeros((n,), np.float32)


def coefficient_fun(x_one):
    for i in range(m):
        for j in range(n):
            x_one[i][j] = np.math.pow(x_one[i][j], i/(m - 1))


coefficient_fun(x_coef)
x_coef = np.array(x_coef, np.float32)

x_cur_gpu = cuda.mem_alloc(x_cur.nbytes)
x_cur_sqr_gpu = cuda.mem_alloc(x_cur_sqr.nbytes)
x_coef_gpu = cuda.mem_alloc(x_coef.nbytes)
y_one_gpu = cuda.mem_alloc(y_one.nbytes)
y_cur_gpu = cuda.mem_alloc(y_cur.nbytes)


cuda.memcpy_htod(x_cur_gpu, x_cur)
cuda.memcpy_htod(x_cur_sqr_gpu, x_cur_sqr)
cuda.memcpy_htod(x_coef_gpu, x_coef)
cuda.memcpy_htod(y_one_gpu, y_one)
cuda.memcpy_htod(y_cur_gpu, y_cur)

fun_pso_str = """
__global__ void fun_sqr(float *x, float *x_sqr)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_sqr[tx] = x[tx] * x[tx];
}
__global__ void fun_multi(float * x_sqr, float * x_coef, float * y_one)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y_one[tx] = x_coef[tx] * x_sqr[tx];
}
__device__ void fun_add_line(float *x, float *y)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y[tx] += x[tx];
}
__global__ void fun_y(float x_two[][%(n)s], float * y_cur)
{
    for(int line = 0; line < %(m)s; line++)
    {
        fun_add_line(x_two[line], y_cur);
    }
}
"""

fun_pso_str = fun_pso_str % {'n': n, 'm': m, 'pow': pow, }
mod_fun_obj = SourceModule(fun_pso_str)

fun_sqr = mod_fun_obj.get_function('fun_sqr')   # 矩阵元素平方
fun_multi = mod_fun_obj.get_function('fun_multi')
fun_y = mod_fun_obj.get_function('fun_y')

print("最初产生的 x_cur: \n", x_cur)

fun_sqr(x_cur_gpu, x_cur_sqr_gpu,  grid=grids, block=(n, 1, 1))
fun_multi(x_cur_sqr_gpu, x_coef_gpu, y_one_gpu, grid=grids, block=(n, 1, 1))
fun_y(y_one_gpu, y_cur_gpu, grid=grids, block=(bn, 1, 1))

cuda.memcpy_dtoh(x_cur_sqr, x_cur_sqr_gpu)
print("GPU 上计算的 x_cur_sqr: \n", x_cur_sqr)
cuda.memcpy_dtoh(x_coef, x_coef_gpu)
print("GPU 上计算的 x_coef: \n", x_coef)
cuda.memcpy_dtoh(y_one, y_one_gpu)
print("GPU 上计算的 y_one: \n", y_one)
cuda.memcpy_dtoh(y_cur, y_cur_gpu)
print("GPU 上计算的 y_cur: \n", y_cur)