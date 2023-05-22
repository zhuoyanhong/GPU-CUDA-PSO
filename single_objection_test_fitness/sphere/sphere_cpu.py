import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit


np.random.seed(42)

m = 2 ** 3   # 维度
n = 2 ** 12  # 粒子数  n = bm * bn
bm = 2 ** 8
bn = 2 ** 2
grids = (2 ** 8, 1, 1)

iter_num = 2
rand_trans_steps = 2 * (10 ** 3)

w = 0.4
c1, c2 = np.float32(2.), np.float32(2.)
r1, r2 = np.random.random((2, rand_trans_steps, 1, n)).astype(np.float32)
offset_rand = r1.nbytes / rand_trans_steps * m

val_max = 0.5  # 速度最值
x_var_max = 10  # 位置最值

start = cuda.Event()
end = cuda.Event()

x_cur = np.random.uniform(-x_var_max, x_var_max, (m, n)).astype(np.float32)

y_cur = np.zeros((n,), np.float32)  # 当前适应度值矩阵


def fitness_func(H, y_cur):
    for j in range(n):
        sum = 0
        for i in range(m):
            sum += H[i][j]**2
        y_cur[j] = sum


if __name__ == '__main__':
    start.record()  # start timing
    start.synchronize()
    print(x_cur)
    fitness_func(x_cur, y_cur)
    print(y_cur)
    end.record()  # end timing
    # calculate the run length
    end.synchronize()
    secs = start.time_till(end) * 1e-3
    print('CPU 总时间： ', secs, '\n')

    # x = np.random.randint(1, 5, (n, m)).astype(np.float32)
    # print(x, '\n')
    # print(x[:1], '\n')
    # print(x[:1]**2, '\n')
    # y = fitness_func(x)
    # print(y)
