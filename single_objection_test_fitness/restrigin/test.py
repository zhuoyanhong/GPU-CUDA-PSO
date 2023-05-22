import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

m = 3
n = 4
a = 10
pi = np.pi


x_cur = np.random.uniform(1, 10, (m, n)).astype(np.float32)
x_cur_gpu = cuda.mem_alloc(x_cur.nbytes)
cuda.memcpy_htod(x_cur_gpu, x_cur)

y_temp = np.zeros((m, n), np.float32)
y_temp_gpu = cuda.mem_alloc(y_temp.nbytes)
cuda.memcpy_htod(y_temp_gpu, y_temp)


fun_pso_str = """
__global__ void fun_cos(float *x, float *y_temp)
{
    int tx = blockIdx.x *blockDim.x + threadIdx.x;
    y_temp[tx] = -%(a)s * cos(2 * %(pi)s * x[tx]) + %(a)s;
}
"""

fun_pso_str = fun_pso_str % {'a': a, 'pi': pi, }
mod_fun_obj = SourceModule(fun_pso_str)

fun_cos = mod_fun_obj.get_function('fun_cos')


print(x_cur)
fun_cos(x_cur_gpu, y_temp_gpu, grid=(m, 1, 1), block=(n, 1, 1))
cuda.memcpy_dtoh(y_temp, y_temp_gpu)
print(y_temp)






