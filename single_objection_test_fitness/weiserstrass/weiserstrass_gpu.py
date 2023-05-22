import numpy as np
import gc
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit


np.random.seed(42)

m = 2 ** 2   # 维度
n = 2 ** 3  # 粒子数  n = bm * bn
bm = 2 ** 2
bn = 2 ** 1
grids = (2 ** 2, 1, 1)

x_var_max = 100  # 位置最值
val_max = 10       # 速度最值

a = 0.5
b = 3
k_max = 20
pi_2 = 2 * np.pi


def fun_after():
    sum = 0
    for k in range(k_max):
        sum += (a ** k) * np.cos(pi_2 * 0.5 * (b ** k))
    y = m * sum
    return y


y_after = fun_after()
print("函数的后缀项: ", y_after)

a_cpu = a * np.ones((m, n), np.float32)
b_cpu = b * np.ones((m, n), np.float32)
a_k_cpu = np.zeros((m, n), np.float32)
b_k_cpu = np.zeros((m, n), np.float32)
x_cur = np.random.uniform(-x_var_max, x_var_max, (m, n)).astype(np.float32)
print("初始产生的 x_cur: \n", x_cur)
x_add = np.zeros((m, n), np.float32)
x_cur_b = np.zeros((m, n), np.float32)
x_cos = np.zeros((m, n), np.float32)
x_cos_a = np.zeros((m, n), np.float32)
x_sum_k = np.zeros((m, n), np.float32)
y_cur = np.zeros((n, ), np.float32)

a_gpu = cuda.mem_alloc(a_cpu.nbytes)
b_gpu = cuda.mem_alloc(b_cpu.nbytes)
a_k_gpu = cuda.mem_alloc(a_k_cpu.nbytes)
b_k_gpu = cuda.mem_alloc(b_k_cpu.nbytes)
x_cur_gpu = cuda.mem_alloc(x_cur.nbytes)
x_add_gpu = cuda.mem_alloc(x_add.nbytes)
x_cur_b_gpu = cuda.mem_alloc(x_cur_b.nbytes)
x_cos_gpu = cuda.mem_alloc(x_cos.nbytes)
x_cos_a_gpu = cuda.mem_alloc(x_cos_a.nbytes)
x_sum_k_gpu = cuda.mem_alloc(x_sum_k.nbytes)
y_cur_gpu = cuda.mem_alloc(y_cur.nbytes)

cuda.memcpy_htod(a_gpu, a_cpu)
cuda.memcpy_htod(b_gpu, b_cpu)
cuda.memcpy_htod(a_k_gpu, a_k_cpu)
cuda.memcpy_htod(b_k_gpu, b_k_cpu)
cuda.memcpy_htod(x_cur_gpu, x_cur)
cuda.memcpy_htod(x_add_gpu, x_add)
cuda.memcpy_htod(x_cur_b_gpu, x_cur_b)
cuda.memcpy_htod(x_cos_gpu, x_cos)
cuda.memcpy_htod(x_cos_a_gpu, x_cos_a)
cuda.memcpy_htod(x_sum_k_gpu, x_sum_k)
cuda.memcpy_htod(y_cur_gpu, y_cur)

fun_pso_str = """
__global__ void fun_x_add(float * x_cur, float * x_0)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_0[tx] = x_cur[tx] + 0.5;
}
__global__ void power_ab(float * b, int k, float * b_k)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    b_k[tx] = pow(b[tx], k);
}
__global__ void fun_cos(float * b, float * x_0, float *x_cos)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_cos[tx] = cos(b[tx] * x_0[tx] * %(pi_2)s);
}
__global__ void fun_cos_a(float * x_cos, float * a, float * x_cos_a)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_cos_a[tx] = x_cos[tx] * a[tx];
}
__global__ void fun_sum_k(float * x_cos_a, float * x_sum_k)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_sum_k[tx] += x_cos_a[tx];
}
__device__ void fun_add_line(float *x, float *y)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y[tx] += x[tx];
}
__global__ void fun_y(float x_sum_k[][%(n)s], float *y_cur)
{
    for(int line = 0; line < %(m)s; line++)
    {
        fun_add_line(x_sum_k[line], y_cur);
    }
}
__global__ void fun_result(float * y_cur)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y_cur[tx] -= %(y_after)s;
}

"""

fun_pso_str = fun_pso_str % {'n': n, 'm': m, 'pi_2': pi_2, 'y_after': y_after}
mod_fun_obj = SourceModule(fun_pso_str)


fun_x_add = mod_fun_obj.get_function('fun_x_add')
power_ab = mod_fun_obj.get_function('power_ab')
fun_cos = mod_fun_obj.get_function('fun_cos')
fun_cos_a = mod_fun_obj.get_function('fun_cos_a')
fun_sum_k = mod_fun_obj.get_function('fun_sum_k')
fun_y = mod_fun_obj.get_function('fun_y')
fun_result = mod_fun_obj.get_function('fun_result')

fun_x_add(x_cur_gpu, x_add_gpu, grid=grids, block=(n, 1, 1))
for k in range(k_max):
    power_ab(a_gpu, np.int32(k), a_k_gpu, grid=grids, block=(n, 1, 1))
    power_ab(b_gpu, np.int32(k), b_k_gpu, grid=grids, block=(n, 1, 1))
    fun_cos(b_k_gpu, x_add_gpu, x_cos_gpu, grid=grids, block=(n, 1, 1))
    fun_cos_a(x_cos_gpu, a_k_gpu, x_cos_a_gpu, grid=grids, block=(n, 1, 1))
    fun_sum_k(x_cos_a_gpu, x_sum_k_gpu, grid=grids, block=(n, 1, 1))
fun_y(x_sum_k_gpu, y_cur_gpu, grid=grids, block=(bn, 1, 1))
fun_result(y_cur_gpu, grid=grids, block=(bn, 1, 1))

# cuda.memcpy_dtoh(x_cur, x_cur_gpu)
# cuda.memcpy_dtoh(x_add, x_add_gpu)
# print("GPU 计算后的 x_cur: \n", x_cur)
# print("GPU 计算后的 x_add: \n", x_add)
#
#
# print("CPU 初始产生的 b_cpu :\n", b_cpu)
# cuda.memcpy_dtoh(b_cpu, b_gpu)
# cuda.memcpy_dtoh(x_cos, x_cos_gpu)
# print("GPU 计算后的 b_cpu: \n", b_cpu)
# print("GPU 计算后的 x_cos: \n", x_cos)
#
#
# cuda.memcpy_dtoh(x_cos_a, x_cos_a_gpu)
# print("GPU 计算后的 x_cos_a: \n", x_cos_a)

cuda.memcpy_dtoh(x_sum_k, x_sum_k_gpu)
print("GPU 计算后的 x_sum_k: \n", x_sum_k)
cuda.memcpy_dtoh(y_cur, y_cur_gpu)
print("GPU 计算后的 y_cur: \n", y_cur)
