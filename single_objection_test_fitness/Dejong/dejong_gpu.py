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

w = 0.8
c1, c2 = np.float32(2.), np.float32(2.)
r1, r2 = np.random.random((2, rand_trans_steps, 1, n)).astype(np.float32)
offset_rand = r1.nbytes / rand_trans_steps * m

val_max = 0.3  # 速度最值
x_var_max = 1  # 位置最值

start = cuda.Event()
end = cuda.Event()

# 在cpu上产生变量
par_w = w * np.ones((m, n), np.float32)  # 惯性权重数组
par_cr1 = c1 * r1 * np.ones((rand_trans_steps, m, n), np.float32)  # 社会因子与学习因子数组
par_cr2 = c2 * r2 * np.ones((rand_trans_steps, m, n), np.float32)
x_max = x_var_max * np.ones((m, n), np.float32)  # 位置最值数组
v_max = val_max * np.ones((m, n), np.float32)  # 速度最值数组
v_cur = np.zeros((m, n), np.float32)  # 当前速度矩阵
v_pre = np.random.uniform(-val_max, val_max, (m, n)).astype(np.float32)  # 更新速度矩阵
x_cur = np.random.uniform(-x_var_max, x_var_max, (m, n)).astype(np.float32)
x_exp = np.zeros((m, n), np.float32)
x_abs = np.zeros((m, n), np.float32)
y_temp = np.zeros((m, n), np.float32)
y_cur = np.zeros((n, ), np.float32)


def fun_exp(x_exp_0):
    for j in range(n):
        for i in range(m):
            x_exp_0[i][j] = i + 2


fun_exp(x_exp)
x_exp = np.array(x_exp, np.float32)


par_w_gpu = cuda.mem_alloc(par_w.nbytes)
par_cr1_gpu = cuda.mem_alloc(par_cr1.nbytes)
par_cr2_gpu = cuda.mem_alloc(par_cr2.nbytes)
x_max_gpu = cuda.mem_alloc(x_max.nbytes)
v_max_gpu = cuda.mem_alloc(v_max.nbytes)
v_cur_gpu = cuda.mem_alloc(v_cur.nbytes)
v_pre_gpu = cuda.mem_alloc(v_pre.nbytes)
x_cur_gpu = cuda.mem_alloc(x_cur.nbytes)
x_exp_gpu = cuda.mem_alloc(x_exp.nbytes)
x_abs_gpu = cuda.mem_alloc(x_abs.nbytes)
y_temp_gpu = cuda.mem_alloc(y_temp.nbytes)
y_cur_gpu = cuda.mem_alloc(y_cur.nbytes)

# 将数据从host拷贝到device
cuda.memcpy_htod(par_w_gpu, par_w)
cuda.memcpy_htod(par_cr1_gpu, par_cr1)
cuda.memcpy_htod(par_cr2_gpu, par_cr2)
cuda.memcpy_htod(x_max_gpu, x_max)
cuda.memcpy_htod(v_max_gpu, v_max)
cuda.memcpy_htod(v_cur_gpu, v_cur)
cuda.memcpy_htod(v_pre_gpu, v_pre)
cuda.memcpy_htod(x_cur_gpu, x_cur)
cuda.memcpy_htod(x_exp_gpu, x_exp)
cuda.memcpy_htod(x_abs_gpu, x_abs)
cuda.memcpy_htod(y_temp_gpu, y_temp)
cuda.memcpy_htod(y_cur_gpu, y_cur)

fun_pso_str = """
__device__ void fun_abs(float *x_cur, float *x_abs)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_abs[tx] = abs(x_cur[tx]);
}
__global__ void fun_temp(float *x_cur, float *x_abs, float *x_exp, float *y_temp)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    fun_abs(x_cur, x_abs);
    y_temp[tx] = pow(x_abs[tx], x_exp[tx]);
}
__device__ void fun_add_line(float *x, float *y)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y[tx] += x[tx];
}
__global__ void fun_y(float y_temp[][%(n)s], float *y_cur)
{
    for(int line = 0; line < %(m)s; line++)
    {
        fun_add_line(y_temp[line], y_cur);
    }
}
"""

fun_pso_str = fun_pso_str % {'n': n, 'm': m, }

mod_fun_obj = SourceModule(fun_pso_str)

fun_temp = mod_fun_obj.get_function('fun_temp')
fun_y = mod_fun_obj.get_function('fun_y')

fun_temp(x_cur_gpu, x_abs_gpu, x_exp_gpu, y_temp_gpu, grid=grids, block=(n, 1, 1))
fun_y(y_temp_gpu, y_cur_gpu, grid=grids, block=(bn, 1, 1))

cuda.memcpy_dtoh(y_cur, y_cur_gpu)
print("GPU 上计算的 y_cur: \n", y_cur)



