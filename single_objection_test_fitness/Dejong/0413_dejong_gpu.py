import numpy as np
import gc
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import time

np.random.seed(42)

m = 2 ** 4   # 维度
n = 2 ** 9  # 粒子数  n = bm * bn
bm = 2 ** 4
bn = 2 ** 5
grids = (2 ** 4, 1, 1)

iter_num = 2000
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
x_pre = np.copy(x_cur)  # 更新位置矩阵
x_pbest = np.copy(x_cur)  # 个体最优位置矩阵
x_gbest = np.zeros((m, n), np.float32)  # 全局最优位置矩阵
x_cur_sqr = np.zeros((m, n), np.float32)
y_cur = np.zeros((n,), np.float32)  # 当前适应度值矩阵
y_pre = np.zeros((n,), np.float32)  # 更新适应度值矩阵
y_pbest_flag_0 = np.zeros((n,), np.float32)  # 个体最优适应度值矩阵
y_pbest_flag_00 = np.zeros((n,), np.float32)
y_pbest_flag_1 = np.zeros((n,), np.float32)
y_pbest_flag_2 = np.zeros((n,), np.float32)
y_pbest_flag_4 = np.zeros((n,), np.float32)
y_gbest = np.zeros((1,), np.float32)  # 全局最优适应度值数组
y_gbest_index = np.zeros((1,), np.int32)  # 全局最优适应度值索引


def fun_exp(x_exp_0):
    for j in range(n):
        for i in range(m):
            x_exp_0[i][j] = i + 2


fun_exp(x_exp)
x_exp= np.array(x_exp, np.float32)


# 在device上设置相应内存
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
x_pre_gpu = cuda.mem_alloc(x_pre.nbytes)
x_pbest_gpu = cuda.mem_alloc(x_pbest.nbytes)
x_gbest_gpu = cuda.mem_alloc(x_gbest.nbytes)
x_cur_sqr_gpu = cuda.mem_alloc(x_cur_sqr.nbytes)
y_cur_gpu = cuda.mem_alloc(y_cur.nbytes)
y_pre_gpu = cuda.mem_alloc(y_pre.nbytes)
y_pbest_flag_gpu_0 = cuda.mem_alloc(y_pbest_flag_0.nbytes)
y_pbest_flag_gpu_00 = cuda.mem_alloc(y_pbest_flag_00.nbytes)
y_pbest_flag_gpu_1 = cuda.mem_alloc(y_pbest_flag_1.nbytes)
y_pbest_flag_gpu_2 = cuda.mem_alloc(y_pbest_flag_2.nbytes)
y_pbest_flag_gpu_4 = cuda.mem_alloc(y_pbest_flag_4.nbytes)
y_gbest_gpu = cuda.mem_alloc(y_gbest.nbytes)
y_gbest_index_gpu = cuda.mem_alloc(y_gbest_index.nbytes)

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
cuda.memcpy_htod(x_pre_gpu, x_pre)
cuda.memcpy_htod(x_pbest_gpu, x_pbest)
cuda.memcpy_htod(x_gbest_gpu, x_gbest)
cuda.memcpy_htod(x_cur_sqr_gpu, x_cur_sqr)
cuda.memcpy_htod(y_cur_gpu, y_cur)
cuda.memcpy_htod(y_pre_gpu, y_pre)
cuda.memcpy_htod(y_pbest_flag_gpu_0, y_pbest_flag_0)
cuda.memcpy_htod(y_pbest_flag_gpu_00, y_pbest_flag_00)
cuda.memcpy_htod(y_pbest_flag_gpu_1, y_pbest_flag_1)
cuda.memcpy_htod(y_pbest_flag_gpu_2, y_pbest_flag_2)
cuda.memcpy_htod(y_pbest_flag_gpu_4, y_pbest_flag_4)
cuda.memcpy_htod(y_gbest_gpu, y_gbest)
cuda.memcpy_htod(y_gbest_index_gpu, y_gbest_index)

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
__global__ void fun_init_y(float *source, float *dest)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    dest[tx] = source[tx];
}
__global__ void fun_comp_pbest_y(float *pre, float *cur, float *dest)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    dest[tx] = (pre[tx] < cur[tx]) ? 1 : 0;
}

__device__ void fun_copy_line(float *x, float *y)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y[tx] = x[tx];
}
__global__ void fun_exp_pbest_flag(float *source, float dest[][%(n)s])
{
    for(int line = 0; line < %(m)s; line++)
    {
        fun_copy_line(source, dest[line]);
    }
}
__global__ void fun_pbest_x(float *pre, float *cur, float *flag, float *pbest)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    pbest[tx] = (flag[tx] > 0.5) ? pre[tx] : cur[tx];
}
__global__ void fun_pbest_y(float *pre, float *cur, float *flag)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    flag[tx] = (flag[tx] > 0.5) ? pre[tx] : cur[tx];
}


__global__ void copy_arr1(float * arr1, float * arr1_0)
{   
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    arr1_0[tx] = arr1[tx];
}
__device__ void copy_arr(float * arr1, float * arr2)
{   
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    arr2[tx] = arr1[tx];

}
__device__ void distribute(float * arr3, float * p, int b)
{   
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    if (tx < b)
    {
        arr3[tx] = p[tx];
    }
}
__device__ void comp(float * arr1, float * arr2, float * arr3, int b)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    if (tx < b)
    {
        if( arr2[tx] > arr3[tx] )
        { 
            arr1[tx] = arr3[tx];
        }
        else
        {
            arr1[tx] = arr2[tx];
        }
    }
}
__global__ void fun_comp_gbest_y
(float * arr1, float * arr2, float * arr3, float * arr4)
{   
    float * p;
    int b;
    for(int i = 1; i < %(n)s; i *= 2)
    {
        b = %(n)s/(2 * i);
        p = arr1 + b;
        copy_arr(arr1, arr2);
        distribute(arr3, p, b);
        comp(arr1, arr2, arr3, b);
        copy_arr(arr4, arr2);
        copy_arr(arr4, arr3);
    }
}
__global__ void fun_gbest_y(float * arr1_0, float * y_gbest)
{   
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y_gbest[tx] = arr1_0[tx];
}
__global__ void fun_gbest_y_index(float * arr1_0, float * y_gbest, int * index)
{
    float * w;
    for(int steps = 0; steps < %(n)s; steps++)
    {   
        w = arr1_0 + steps;
        if(y_gbest[0] == *w){index[0] = steps;}
    }
}

__device__ void fun_gbest_x_copy(float x_pbest, float * x_gbest)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_gbest[tx] = x_pbest;
}
__global__ void fun_gbest_x(
int *gbest_index, float (*x_pbest)[%(n)s], float (*x_gbest)[%(n)s]
)
{
    for(int line=0; line < %(m)s; line++)
    {
        fun_gbest_x_copy(x_pbest[line][gbest_index[0]], x_gbest[line]);
    }
}
__global__ void fun_vol_update(
float *v_cur, float *v_pre, float *x_cur,
float *pbest, float *gbest,
float *w, float *cr1, float *cr2)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    v_cur[tx] = w[tx] * v_pre[tx] 
                + cr1[tx] * (pbest[tx] - x_cur[tx])
                + cr2[tx] * (gbest[tx] - x_cur[tx]);
}
__global__ void fun_copy(float source[%(n)s], float dest[%(n)s])
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    dest[tx] = source[tx];
}
__global__ void fun_x_update(float *x_pre, float *x_cur, float *v_cur)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x_cur[tx] = x_pre[tx] + v_cur[tx];
}
__global__ void fun_flush_zeros(float *x)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    x[tx] = 0;
}
__global__ void fun_clip(float *x, float *max)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if(x[tx] > max[tx]) x[tx] = max[tx];
    if(x[tx] < -max[tx]) x[tx] = -max[tx];
}

"""
fun_pso_str = fun_pso_str % {'n': n, 'm': m, }

mod_fun_obj = SourceModule(fun_pso_str)

fun_temp = mod_fun_obj.get_function('fun_temp')
fun_y = mod_fun_obj.get_function('fun_y')
fun_init_y = mod_fun_obj.get_function('fun_init_y')  # 初始化适应度矩阵
fun_comp_pbest_y = mod_fun_obj.get_function('fun_comp_pbest_y')
fun_exp_pbest_flag = mod_fun_obj.get_function('fun_exp_pbest_flag')  #
fun_pbest_x = mod_fun_obj.get_function('fun_pbest_x')  # 更新个体最优位置矩阵
fun_pbest_y = mod_fun_obj.get_function('fun_pbest_y')  # 更新个体最优适应度矩阵
fun_copy_arr1 = mod_fun_obj.get_function("copy_arr1")
fun_comp_gbest_y = mod_fun_obj.get_function("fun_comp_gbest_y")
fun_gbest_y = mod_fun_obj.get_function("fun_gbest_y")
fun_gbest_y_index = mod_fun_obj.get_function("fun_gbest_y_index")
fun_gbest_x = mod_fun_obj.get_function('fun_gbest_x')  # 全局最优位置
fun_vol_update = mod_fun_obj.get_function('fun_vol_update')  # 速度更新
fun_copy = mod_fun_obj.get_function('fun_copy')
fun_x_update = mod_fun_obj.get_function('fun_x_update')  # 位置更新
fun_flush_zeros = mod_fun_obj.get_function('fun_flush_zeros')
fun_clip = mod_fun_obj.get_function('fun_clip')

# 计算适应度值  y_cur
fun_temp(x_cur_gpu, x_abs_gpu, x_exp_gpu, y_temp_gpu, grid=grids, block=(n, 1, 1))
fun_y(y_temp_gpu, y_cur_gpu, grid=grids, block=(bn, 1, 1))
cuda.memcpy_dtoh(y_cur, y_cur_gpu)
# print("GPU 上计算的 y_cur: \n", y_cur)
# 初始化适应度值 y_pre
fun_init_y(y_cur_gpu, y_pre_gpu, grid=grids, block=(bn, 1, 1))

start.synchronize()
start_time = time.time()
for step in range(iter_num):

    rand_flush_flag = step % rand_trans_steps
    if step and (rand_flush_flag == 0):
        r1, r2 = np.random.random((2, rand_trans_steps, 1, n)).astype(
            np.float32)
        par_cr1 = c1 * r1 * np.ones((rand_trans_steps, m, n), np.float32)
        par_cr2 = c2 * r2 * np.ones((rand_trans_steps, m, n), np.float32)
        cuda.memcpy_htod(par_cr1_gpu, par_cr1)
        cuda.memcpy_htod(par_cr2_gpu, par_cr2)
        del r1, r2, par_cr1, par_cr2
        gc.collect()

    fun_comp_pbest_y(y_pre_gpu, y_cur_gpu, y_pbest_flag_gpu_0,
                     grid=grids, block=(bn, 1, 1))
    fun_exp_pbest_flag(y_pbest_flag_gpu_0, x_cur_sqr_gpu,
                       grid=grids, block=(bn, 1, 1))
    fun_pbest_x(x_pre_gpu, x_cur_gpu, x_cur_sqr_gpu, x_pbest_gpu,
                grid=grids, block=(bm * bn, 1, 1))
    fun_pbest_y(y_pre_gpu, y_cur_gpu, y_pbest_flag_gpu_0,
                grid=grids, block=(bn, 1, 1))

    fun_copy_arr1(y_pbest_flag_gpu_0, y_pbest_flag_gpu_00, grid=grids,
                  block=(bn, 1, 1))
    fun_comp_gbest_y(y_pbest_flag_gpu_00, y_pbest_flag_gpu_1,
                     y_pbest_flag_gpu_2, y_pbest_flag_gpu_4,
                     grid=grids, block=(bn, 1, 1))
    fun_gbest_y(y_pbest_flag_gpu_00, y_gbest_gpu,
                grid=(1, 1, 1), block=(1, 1, 1))
    fun_gbest_y_index(y_pbest_flag_gpu_0, y_gbest_gpu, y_gbest_index_gpu,
                      grid=grids, block=(bn, 1, 1))

    fun_gbest_x(y_gbest_index_gpu, x_pbest_gpu, x_gbest_gpu,
                grid=grids, block=(bn, 1, 1))

    fun_vol_update(
        v_cur_gpu, v_pre_gpu, x_cur_gpu,
        x_pbest_gpu, x_gbest_gpu,
        par_w_gpu,
        np.intp(par_cr1_gpu.__int__() + rand_flush_flag * offset_rand),
        np.intp(par_cr2_gpu.__int__() + rand_flush_flag * offset_rand),
        grid=grids,
        block=(n, 1, 1),
    )
    fun_clip(v_cur_gpu, v_max_gpu, grid=grids, block=(n, 1, 1))

    fun_copy(x_cur_gpu, x_pre_gpu, grid=grids, block=(n, 1, 1))
    fun_copy(v_cur_gpu, v_pre_gpu, grid=grids, block=(n, 1, 1))
    fun_copy(y_cur_gpu, y_pre_gpu, grid=grids, block=(bn, 1, 1))
    fun_flush_zeros(y_cur_gpu, grid=grids, block=(bn, 1, 1))
    fun_flush_zeros(x_abs_gpu, grid=grids, block=(n, 1, 1))
    fun_flush_zeros(y_temp_gpu, grid=grids, block=(n, 1, 1))

    fun_x_update(x_pre_gpu, x_cur_gpu, v_cur_gpu,
                 grid=grids, block=(n, 1, 1))
    fun_clip(x_cur_gpu, x_max_gpu, grid=grids, block=(n, 1, 1))

    fun_temp(x_cur_gpu, x_abs_gpu, x_exp_gpu, y_temp_gpu, grid=grids, block=(n, 1, 1))
    fun_y(y_temp_gpu, y_cur_gpu, grid=grids, block=(bn, 1, 1))

end.synchronize()
end_time = time.time()
print("总用时： ", (end_time - start_time))
cuda.memcpy_dtoh(y_cur, y_cur_gpu)
# print("GPU 上计算的 y_cur: \n", y_cur)
print("GPU 计算的 min(y_cur)：", np.min(y_cur))
cuda.memcpy_dtoh(y_gbest, y_gbest_gpu)
print("GPU 计算的 y_gbest: ", y_gbest)
