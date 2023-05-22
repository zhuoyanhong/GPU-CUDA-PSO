import numpy as np
import gc
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import time

np.random.seed(42)

m = 2 ** 2   # 维度
n = 2 ** 9  # 粒子数  n = bm * bn
bm = 2 ** 2
bn = 2 ** 7
grids = (2 ** 2, 1, 1)

iter_num = 1
rand_trans_steps = 2 * (10 ** 3)

w = 0.3
c1, c2 = np.float32(2.), np.float32(2.)
r1, r2 = np.random.random((2, rand_trans_steps, 1, n)).astype(np.float32)
offset_rand = r1.nbytes / rand_trans_steps * m

val_max = 0.5  # 速度最值
x_var_max = 13  # 位置最大值
x_var_min = 3
c = 2 / 3
d = 1.21598 * m

start = cuda.Event()
end = cuda.Event()

# 在cpu上产生变量
par_w = w * np.ones((m, n), np.float32)  # 惯性权重数组
par_cr1 = c1 * r1 * np.ones((rand_trans_steps, m, n), np.float32)  # 社会因子与学习因子数组
par_cr2 = c2 * r2 * np.ones((rand_trans_steps, m, n), np.float32)
x_max = x_var_max * np.ones((m, n), np.float32)  # 位置最值数组
x_min = x_var_min * np.ones((m, n), np.float32)  # 位置最小值数组
v_max = val_max * np.ones((m, n), np.float32)  # 速度最值数组
v_cur = np.zeros((m, n), np.float32)  # 当前速度矩阵
v_pre = np.random.uniform(-val_max, val_max, (m, n)).astype(np.float32)  # 更新速度矩阵
x_cur = np.random.uniform(x_var_min, x_var_max, (m, n)).astype(np.float32)
x_sin_one = np.zeros((m, n), np.float32)
x_sin_two = np.zeros((m, n), np.float32)
y_one = np.zeros((m, n), np.float32)
y_cur = np.zeros((n,), np.float32)  # 当前适应度值矩阵

# 在device上设置相应内存
par_w_gpu = cuda.mem_alloc(par_w.nbytes)
par_cr1_gpu = cuda.mem_alloc(par_cr1.nbytes)
par_cr2_gpu = cuda.mem_alloc(par_cr2.nbytes)
x_max_gpu = cuda.mem_alloc(x_max.nbytes)
x_min_gpu = cuda.mem_alloc(x_min.nbytes)
v_max_gpu = cuda.mem_alloc(v_max.nbytes)
v_cur_gpu = cuda.mem_alloc(v_cur.nbytes)
v_pre_gpu = cuda.mem_alloc(v_pre.nbytes)
x_cur_gpu = cuda.mem_alloc(x_cur.nbytes)
x_sin_one_gpu = cuda.mem_alloc(x_sin_one.nbytes)
x_sin_two_gpu = cuda.mem_alloc(x_sin_two.nbytes)
y_one_gpu = cuda.mem_alloc(y_one.nbytes)
y_cur_gpu = cuda.mem_alloc(y_cur.nbytes)

# 将数据从host拷贝到device
cuda.memcpy_htod(par_w_gpu, par_w)
cuda.memcpy_htod(par_cr1_gpu, par_cr1)
cuda.memcpy_htod(par_cr2_gpu, par_cr2)
cuda.memcpy_htod(x_max_gpu, x_max)
cuda.memcpy_htod(x_min_gpu, x_min)
cuda.memcpy_htod(v_max_gpu, v_max)
cuda.memcpy_htod(v_cur_gpu, v_cur)
cuda.memcpy_htod(v_pre_gpu, v_pre)
cuda.memcpy_htod(x_cur_gpu, x_cur)
cuda.memcpy_htod(x_sin_one_gpu, x_sin_one)
cuda.memcpy_htod(x_sin_two_gpu, x_sin_two)
cuda.memcpy_htod(y_one_gpu, y_one)
cuda.memcpy_htod(y_cur_gpu, y_cur)

fun_pso_str = """
__global__ void fun_sin_one(float *x_cur, float *x_sin_one)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_sin_one[tx] = sin(x_cur[tx]);
}
__global__ void fun_sin_two(float *x_cur, float *x_sin_two)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_sin_two[tx] = sin(x_cur[tx] * %(c)s);
}
__global__ void fun_y_temp(float * x_sin_one, float *x_sin_two, float *y_one)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y_one[tx] = x_sin_one[tx] + x_sin_two[tx];
}
__device__ void fun_add_line(float *x, float *y)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y[tx] += x[tx];
}
__global__ void fun_y(float y_temp[][%(n)s], float *y_cur)
{   
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    for(int line = 0; line < %(m)s; line++)
    {
        fun_add_line(y_temp[line], y_cur);
    }
    y_cur[tx] += %(d)s;
}
"""

fun_pso_str = fun_pso_str % {'n': n, 'm': m, 'c': c, 'd': d, }

mod_fun_obj = SourceModule(fun_pso_str)

fun_sin_one = mod_fun_obj.get_function('fun_sin_one')
fun_sin_two = mod_fun_obj.get_function('fun_sin_two')
fun_y_temp = mod_fun_obj.get_function('fun_y_temp')
fun_y = mod_fun_obj.get_function('fun_y')  # 矩阵列求和

print("初始产生的 x_cur： \n", x_cur)
print("初始产生的 v_pre： \n", v_pre)
# 计算适应度值  y_cur
fun_sin_one(x_cur_gpu, x_sin_one_gpu, grid=grids, block=(n, 1, 1))
fun_sin_two(x_cur_gpu, x_sin_two_gpu, grid=grids, block=(n, 1, 1))
fun_y_temp(x_sin_one_gpu, x_sin_two_gpu, y_one_gpu, grid=grids, block=(n, 1, 1))
fun_y(y_one_gpu, y_cur_gpu, grid=grids, block=(bn, 1, 1))

cuda.memcpy_dtoh(x_sin_one, x_sin_one_gpu)
print('GPU 计算的 x_sin_one: \n', x_sin_one)
cuda.memcpy_dtoh(x_sin_two, x_sin_two_gpu)
print('GPU 计算的 x_sin_two: \n', x_sin_two)
cuda.memcpy_dtoh(y_cur, y_cur_gpu)
print('GPU 计算的 y_cur: \n', y_cur)
# cuda.memcpy_dtoh(x_cur, x_cur_gpu)
# print('GPU 计算的 x_cur: \n', x_cur)

