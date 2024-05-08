import random
import math
import pandas as pd
import numpy as np
import time


start = time.time()
shuju = np.array(pd.read_csv(r'附件1：data_100.csv'))
t = shuju[:, ::2].T  # 通过率
t = list(np.array(t).reshape(1000, ))

c = shuju[:, 1::2].T  # 坏账率
c = list(np.array(c).reshape(1000, ))
a = 1000000
M = 12000
b = 0.08


# 定义适应度函数
def fitness_function(x, t, c):
    cost = 0
    cost = sum([(a * b * t[i] * t[j] * t[k] - a * t[i] * t[j] * t[k] * (b + 1) * (1 / 3) *
                 (c[i] + c[j] + c[k])) * (x[i] * x[j] * x[k])
                for l1 in range(98)
                for i in range(10 * l1, 10 * (l1 + 1))
                for l2 in range(l1 + 1, 99)
                for j in range(10 * l2, 10 * (l2 + 1))
                for l3 in range(l2 + 1, 100)
                for k in range(10 * l3, 10 * (l3 + 1))])

    ystj1 = (sum(x) - 3) ** 2
    ystj2 = sum([x[i] * x[j] for l in range(100)
                 for i in range(10 * l, 10 * (l + 1) - 1)
                 for j in range(10 * l + i + 1, 10 * (l + 1))])
    H = -cost + M * ystj1 + M * ystj2
    return H


# 定义模拟退火算法的主函数
def simulated_annealing(n_iterations, initial_temperature, final_temperature, cooling_rate, n_variables):
    # 初始化当前解和当前目标函数值
    current_solution = [random.randint(0, 0) for i in range(n_variables)]
    # current_solution[random.randint(0, 999)]=1
    current_cost = fitness_function(current_solution, t, c)

    # 初始化最优解和最优目标函数值
    best_solution = current_solution
    best_cost = current_cost

    # 初始化当前温度
    current_temperature = initial_temperature

    # 模拟退火算法的主循环
    for i in range(n_iterations):
        # 生成新解
        new_solution = list(np.zeros(1000))
        new_solution[random.randint(0, 999)] = 1
        new_solution[random.randint(0, 999)] = 1
        new_solution[random.randint(0, 999)] = 1
        # new_solution = [random.randint(0,0) for i in range(n_variables)]

        # 计算新解的目标函数值和约束条件
        new_cost = fitness_function(new_solution, t, c)

        # 判断是否接受新解
        if new_cost < current_cost:
            current_solution = new_solution
            current_cost = new_cost

            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

        else:
            delta = new_cost - current_cost
            probability = math.exp(-delta / current_temperature)

            if random.random() < probability:
                current_solution = new_solution
                current_cost = new_cost

        # 降温
        current_temperature *= cooling_rate

        if current_temperature < final_temperature:
            break

    return best_solution, best_cost


# 调用模拟退火算法求解0-1规划问题
jieguo = []
index = []
for pp in range(10):
    solution, cost = simulated_annealing(n_iterations=10, initial_temperature=100, final_temperature=0.01,
                                         cooling_rate=0.99, n_variables=1000)
    for qq in range(1000):
        if solution[qq] == 1:
            index.append(qq)
            jieguo.append(cost)

for i in range(len(index)):
    if (i > 0) & (i % 3 == 0):
        a = int(i / 3)
        print(index[i - 2], index[i - 1], index[i], jieguo[a])

end = time.time()
print (str(end))
# 输出结果
