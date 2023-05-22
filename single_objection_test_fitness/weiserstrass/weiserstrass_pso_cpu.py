import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False    # 正常显示图像中的负号

np.random.seed(42)

iter_num = 30
rand_trans_steps = 2 * (10 ** 3)


# pso 的参数
dim = 2 ** 6
size = 2 ** 9

x_var_max = 100  # 位置最值
val_max = 10       # 速度最值

a = 0.5
b = 3
k_max = 20
pi_2 = 2 * np.pi

w = 0.5
c1, c2 = np.float32(2.), np.float32(2.)
r1, r2 = np.random.random((2, rand_trans_steps, 1, size)).astype(np.float32)
offset_rand = r1.nbytes / rand_trans_steps * dim

par_w = w * np.ones((dim, size))  # 惯性权重数组
par_cr1 = c1 * r1 * np.ones((rand_trans_steps, dim, size))
par_cr2 = c2 * r2 * np.ones((rand_trans_steps, dim, size))
v_cur = np.random.uniform(-val_max, val_max, size=(dim, size))
x_cur = np.random.uniform(-x_var_max, x_var_max, size=(dim, size))
# print("初始产生的 x_cur: \n", x_cur)
y_temp = np.zeros((dim, size), np.float32)
y_cur = np.zeros((size, ))
y_pre = np.zeros((size, ))
x_gbest = np.zeros((dim, size))
fitness_value_list = []


# 适应度函数
def fitness_fun(x_cur_0, y_cur_0):
    add = 0
    y_temp_0 = np.zeros((dim, size), np.float32)
    for k in range(k_max):
        x_temp_0 = (a ** k) * np.cos(pi_2 * (b ** k) * (x_cur_0 + 0.5))
        y_temp_0 += x_temp_0
        add += (a ** k) * np.cos(pi_2 * 0.5 * (b ** k))
    y = dim * add
    for j in range(size):
        sum = 0
        for i in range(dim):
            sum += y_temp_0[i][j]
        y_cur_0[j] = sum - y


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
    x_pre_0[x_pre_0 < -x_var_max] = -x_var_max
    x_pre_0[x_pre_0 > x_var_max] = x_var_max
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


def pso_cpu(x_cur, v_cur, x_gbest):
    start_time = time.time()
    # 初次计算种群适应度值与全局最优
    fitness_fun(x_cur, y_cur)
    y_pbest = y_cur
    y_gbest_index = y_pbest.argmin()
    y_gbest = y_pbest[y_gbest_index]
    x_pbest = x_cur
    x_gbest = x_gbest_update(x_pbest, y_gbest_index, x_gbest)
    for step in range(iter_num):
        v_pre = v_update(v_cur, x_cur, x_pbest, x_gbest, par_w, par_cr1[step], par_cr2[step])
        x_pre = x_update(x_cur, v_pre)
        fitness_fun(x_pre, y_pre)
        # 更新个体适应度最优与个体位置最优
        pbest_update(y_pre, y_pbest, x_pre, x_pbest)

        # 更新全局适应度最优与位置最优
        y_gbest_index = y_pbest.argmin()
        y_gbest = y_pbest[y_gbest_index]
        x_gbest_update(x_pbest, y_gbest_index, x_gbest)

        x_cur = x_pre
        v_cur = v_pre

        # 记录最优迭代结果
        fitness_value_list.append(y_gbest)
    end_time = time.time()
    print("总用时： ", (end_time - start_time))
    print("CPU 计算的值 min(y_pre): ", min(y_pre))
    print("CPU 计算的值 y_gbest： ", y_gbest)


pso_cpu(x_cur, v_cur, x_gbest)

# # 绘图
# plt.plot(fitness_value_list, color='r')
# plt.title('迭代过程_cpu')
# plt.show()




