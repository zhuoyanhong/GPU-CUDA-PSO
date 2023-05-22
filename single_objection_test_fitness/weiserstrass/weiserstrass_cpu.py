import numpy as np

np.random.seed(42)

iter_num = 1
rand_trans_steps = 2 * (10 ** 3)

m = 2 ** 6   # 维度
n = 2 ** 9  # 粒子数  n = bm * bn
bm = 2 ** 6
bn = 2 ** 3
grids = (2 ** 6, 1, 1)

x_var_max = 100  # 位置最值
val_max = 10       # 速度最值

a = 0.5
b = 3
k_max = 20
pi_2 = 2 * np.pi

w = 0.5
c1, c2 = np.float32(2.), np.float32(2.)
r1, r2 = np.random.random((2, rand_trans_steps, 1, n)).astype(np.float32)
offset_rand = r1.nbytes / rand_trans_steps * m

par_w = w * np.ones((m, n))  # 惯性权重数组
par_cr1 = c1 * r1 * np.ones((rand_trans_steps, m, n))
par_cr2 = c2 * r2 * np.ones((rand_trans_steps, m, n))
v_cur = np.random.uniform(-val_max, val_max, size=(m, n))
x_cur = np.random.uniform(-x_var_max, x_var_max, (m, n)).astype(np.float32)
print("初始产生的 x_cur: \n", x_cur)
y_temp = np.zeros((m, n), np.float32)
y_zero = np.copy(y_temp)
y_cur = np.zeros((n, ), np.float32)


def fitness_fun(x_cur_0, y_cur_0):
    add = 0
    y_temp_0 = np.zeros((m, n), np.float32)
    for k in range(k_max):
        x_temp_0 = (a ** k) * np.cos(pi_2 * (b ** k) * (x_cur_0 + 0.5))
        y_temp_0 += x_temp_0
        add += (a ** k) * np.cos(pi_2 * 0.5 * (b ** k))
    y = m * add
    for j in range(n):
        sum = 0
        for i in range(m):
            sum += y_temp_0[i][j]
        y_cur_0[j] = sum - y


# fitness_fun(x_cur, y_temp, y_cur)
# print("CPU 上计算的 y_cur: \n", y_cur)

for i in range(10):
    fitness_fun(x_cur, y_cur)
    # y_temp = np.zeros((m, n), np.float32)
    print("CPU 上计算的最小值 y_cur: ", y_cur[y_cur.argmin()])
    # print(y_temp)