import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False    # 正常显示图像中的负号

np.random.seed(42)

iter_num = 2
rand_trans_steps = 2 * (10 ** 3)


# pso 的参数
dim = 2 ** 3
size = 2 ** 12
val_max = 1.0
x_max = 30

pi = np.pi
e = np.exp(1)

w = 0.6
c1, c2 = np.float32(2.), np.float32(2.)
r1, r2 = np.random.random((2, rand_trans_steps, 1, size)).astype(np.float32)
offset_rand = r1.nbytes / rand_trans_steps * dim

par_w = w * np.ones((dim, size), np.float32)  # 惯性权重数组
par_cr1 = c1 * r1 * np.ones((rand_trans_steps, dim, size)).astype(np.float32)
par_cr2 = c2 * r2 * np.ones((rand_trans_steps, dim, size)).astype(np.float32)
v_pre = np.random.uniform(-val_max, val_max, (dim, size)).astype(np.float32)
x_cur = np.random.uniform(-x_max, x_max, (dim, size)).astype(np.float32)
x_cur_sqr = np.zeros((dim, size), np.float32)
x_cos = np.zeros((dim, size), np.float32)
x_cos_sum = np.zeros((size, ), np.float32)
x_cos_sum_div = np.zeros((size, ), np.float32)
x_sqr_sum = np.zeros((size, ), np.float32)
x_sqr_sum_div = np.zeros((size, ), np.float32)
x_sqr_sum_div_sqrt = np.zeros((size, ), np.float32)
y_cur = np.zeros((size,), np.float32)  # 当前适应度值矩阵
y_pre = np.zeros((size, ), np.float32)
x_gbest = np.zeros((dim, size), np.float32)
fitness_value_list = []


# 适应度函数
def ackely_fun_two(x_cur, x_cos, x_cos_sum, x_cos_sum_div):
    for j in range(size):
        sum0 = 0
        for i in range(dim):
            x_cos[i][j] = np.cos(2 * pi * x_cur[i][j])
            sum0 += x_cos[i][j]
        x_cos_sum[j] = sum0
        x_cos_sum_div[j] = x_cos_sum[j] / dim


def ackely_fun_one(x_cur, x_cur_sqr, x_sqr_sum, x_sqr_sum_div, x_sqr_sum_div_sqrt):
    for j in range(size):
        sum1 = 0
        for i in range(dim):
            x_cur_sqr[i][j] = x_cur[i][j] ** 2
            sum1 += x_cur_sqr[i][j]
        x_sqr_sum[j] = sum1
        x_sqr_sum_div[j] = x_sqr_sum[j] / dim
        x_sqr_sum_div_sqrt[j] = np.math.sqrt(x_sqr_sum_div[j])


def ackely_fun(x_sqr_sum_div_sqrt, x_cos_sum_div, y_cur):
    for j in range(size):
        y_cur[j] = 20 + e - 20 * np.exp(-0.2 * x_sqr_sum_div_sqrt[j]) - np.exp(x_cos_sum_div[j])


# 速度更新函数
def v_update(v_cur_0, x_cur_0, x_pbest_0, x_gbest_0, par_w_0, par_cr1_0, par_cr2_0):
    v_pre_0 = v_cur_0 * par_w_0\
                    + par_cr1_0 * (x_pbest_0 - x_cur_0) \
                    + par_cr2_0 * (x_gbest_0 - x_cur_0)
    # 防止越界处理
    v_pre_0[v_pre_0 < -val_max] = -val_max
    v_pre_0[v_pre_0 > val_max] = val_max
    return v_pre_0


# 位置更新函数
def x_update(x_cur_0, v_pre_0):
    x_pre_0 = x_cur_0 + v_pre_0
    # 防止越界处理
    x_pre_0[x_pre_0 < -x_max] = -x_max
    x_pre_0[x_pre_0 > x_max] = x_max
    return x_pre_0


def x_gbest_update(x_pbest_0, y_gbest_index_0, x_gbest_0):
    for j in range(size):
        for i in range(dim):
            x_gbest_0[i][j] = x_pbest_0[i][y_gbest_index_0]
    return x_gbest_0


def pbest_update(y_pre_0, y_pbest_0, x_pre_0, x_pbest_0):
    for j in range(size):
        if y_pre_0[j] < y_pbest_0[j]:
            y_pbest_0[j] = y_pre_0[j]
            for i in range(dim):
                x_pbest_0[i][j] = x_pre_0[i][j]


# ackely_fun_two(x_cur, x_cos, x_cos_div, x_cos_sum)
# ackely_fun_one(x_cur, x_cur_sqr, x_sqr_sum, x_sqr_sum_div, x_sqr_sum_div_sqrt)
# ackely_fun(x_sqr_sum_div_sqrt, x_cos_sum, y_cur)
# print('x_cur 的值：\n', x_cur)
# print('y_cur 的值：\n', y_cur)
# print("y_gbest_index: ", y_gbest_index)
# print("y_gbest 的值: ", y_gbest)
# print("x_gbest 的值： \n", x_gbest)


def pso_cpu(x_cur, v_pre, x_gbest):
    start_time = time.time()
    # 初次计算种群适应度值与全局最优
    ackely_fun_two(x_cur, x_cos,  x_cos_sum, x_cos_sum_div,)
    ackely_fun_one(x_cur, x_cur_sqr, x_sqr_sum, x_sqr_sum_div, x_sqr_sum_div_sqrt)
    ackely_fun(x_sqr_sum_div_sqrt, x_cos_sum_div, y_cur)
    # print('y_cur 的值：\n', y_cur)
    y_pbest = y_cur
    y_gbest_index = y_pbest.argmin()
    y_gbest = y_pbest[y_gbest_index]
    x_pbest = x_cur
    x_gbest = x_gbest_update(x_pbest, y_gbest_index, x_gbest)
    for step in range(iter_num):
        v_cur = v_update(v_pre, x_cur, x_pbest, x_gbest, par_w, par_cr1[step], par_cr2[step])
        x_pre = x_update(x_cur, v_cur)
        ackely_fun_two(x_pre, x_cos, x_cos_sum, x_cos_sum_div, )
        ackely_fun_one(x_pre, x_cur_sqr, x_sqr_sum, x_sqr_sum_div, x_sqr_sum_div_sqrt)
        ackely_fun(x_sqr_sum_div_sqrt, x_cos_sum_div, y_pre)
        # 更新个体适应度最优与个体位置最优
        pbest_update(y_pre, y_pbest, x_pre, x_pbest)

        # 更新全局适应度最优与位置最优
        y_gbest_index = y_pbest.argmin()
        y_gbest = y_pbest[y_gbest_index]
        x_gbest_update(x_pbest, y_gbest_index, x_gbest)

        x_cur = x_pre
        v_pre = v_cur

        # 记录最优迭代结果
        fitness_value_list.append(y_gbest)
    end_time = time.time()
    print("总用时： ", (end_time - start_time))
    # print("v_pre 的值： \n", v_pre)
    # print("x_cur 的值： \n", x_cur)
    # print('y_pre 的值：\n', y_pre)
    print("CPU 计算的值 min(y_pre): ", min(y_pre))
    print("CPU 计算的值 y_gbest： ", y_gbest)


pso_cpu(x_cur, v_pre, x_gbest)

# # 绘图
# plt.plot(fitness_value_list, color='r')
# plt.title('迭代过程_cpu')
# plt.show()
# print("par_cr1 的值： \n", par_cr1[0])
# print("par_cr2 的值： \n", par_cr2[0])
# print("v_pre 的值： \n", v_cur)
# print("x_pre 的值： \n", x_cur)
print('y_pre 的值：\n', y_pre)
# print("CPU 计算的值 fitness_value_list： \n", fitness_value_list)
