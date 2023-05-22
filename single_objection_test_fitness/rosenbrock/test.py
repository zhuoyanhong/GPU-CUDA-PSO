import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit


np.random.seed(42)

m = 3   # 维度
n = 4   # 粒子数  n = bm * bn
bm = 2 ** 8
bn = 2 ** 2
grids = (2 ** 8, 1, 1)

iter_num = 2000
rand_trans_steps = 2 * (10 ** 3)

w = 0.4
c1, c2 = np.float32(2.), np.float32(2.)
r1, r2 = np.random.random((2, rand_trans_steps, 1, n)).astype(np.float32)
offset_rand = r1.nbytes / rand_trans_steps * m

a = 100
b = (m-1) * n
val_max = 0.5  # 速度最值
x_var_max = 5  # 位置最值

start = cuda.Event()
end = cuda.Event()

x_cur = np.random.randint(1, 5, (m, n)).astype(np.float32)
x_cur_sqr = np.zeros((m, n), np.float32)
x_one = np.zeros((m, n), np.float32)
x_two = np.zeros((m, n), np.float32)
x_two_0 = np.zeros((m, n), np.float32)
y_cur = np.zeros((n,), np.float32)  # 当前适应度值矩阵

x_cur_gpu = cuda.mem_alloc(x_cur.nbytes)
x_cur_sqr_gpu = cuda.mem_alloc(x_cur_sqr.nbytes)
x_one_gpu = cuda.mem_alloc(x_one.nbytes)
x_two_gpu = cuda.mem_alloc(x_two.nbytes)
x_two_0_gpu = cuda.mem_alloc(x_two_0.nbytes)
y_cur_gpu = cuda.mem_alloc(y_cur.nbytes)

cuda.memcpy_htod(x_cur_gpu, x_cur)
cuda.memcpy_htod(x_cur_sqr_gpu, x_cur_sqr)
cuda.memcpy_htod(x_one_gpu, x_one)
cuda.memcpy_htod(x_two_0_gpu, x_two_0)
cuda.memcpy_htod(x_two_gpu, x_two)
cuda.memcpy_htod(y_cur_gpu, y_cur)

fun_pso_str = """
__global__ void fun_sqr(float *x, float *x_sqr)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_sqr[tx] = x[tx] * x[tx];
}
__device__ void fun_sqr_0(float *x, float *x_sqr)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_sqr[tx] = x[tx] * x[tx];
}
__device__ void distribute_0(float * arr, float * p)
{   
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    if(tx < %(b)s){arr[tx] = p[tx];}
    else{arr[tx] = 0;}
}
__global__ void fun_one(float * x, float * x_sqr, float * x_one)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    float * p;
    p = x + %(n)s;
    distribute_0(x_one, p);
    x_one[tx] = x_sqr[tx] - x_one[tx];
    fun_sqr_0(x_one, x_one);
    x_one[tx] *= %(a)s;
}
__global__ void fun_two(float * x, float * x_two)
{   
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_two[tx] = x[tx] - 1;
    fun_sqr_0(x_two, x_two);
}
__global__ void fun_sum(float * x_one, float * x_two)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_two[tx] += x_one[tx];
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

fun_pso_str = fun_pso_str % {'n': n, 'm': m, 'a': a, 'b': b, }
mod_fun_obj = SourceModule(fun_pso_str)

fun_sqr = mod_fun_obj.get_function('fun_sqr')
fun_one = mod_fun_obj.get_function('fun_one')
fun_two = mod_fun_obj.get_function('fun_two')
fun_sum = mod_fun_obj.get_function('fun_sum')   # 矩阵求和运算
fun_add = mod_fun_obj.get_function('fun_add')   # 矩阵列求和

print("初始位置值: \n", x_cur)
fun_sqr(x_cur_gpu, x_cur_sqr_gpu, grid=grids, block=(n, 1, 1))
fun_one(x_cur_gpu, x_cur_sqr_gpu, x_one_gpu, grid=grids, block=(n, 1, 1))
fun_two(x_cur_gpu, x_two_gpu,  grid=grids, block=(n, 1, 1))
fun_sum(x_one_gpu, x_two_gpu,  grid=grids, block=(n, 1, 1))

fun_add(x_two_gpu, y_cur_gpu, grid=grids, block=(bn, 1, 1))

cuda.memcpy_dtoh(x_one, x_one_gpu)
cuda.memcpy_dtoh(x_cur_sqr, x_cur_sqr_gpu)
cuda.memcpy_dtoh(x_two, x_two_gpu)
cuda.memcpy_dtoh(x_cur, x_cur_gpu)
cuda.memcpy_dtoh(y_cur, y_cur_gpu)
print("GPU 上计算的 x_cur_sqr：\n", x_cur_sqr)
print("GPU 上计算的 x_one： \n", x_one)
print("GPU 上计算的 x_two： \n", x_two)
print("GPU 上计算的 x_cur： \n", x_cur)
print("GPU 上计算的 y_cur： \n", y_cur)


