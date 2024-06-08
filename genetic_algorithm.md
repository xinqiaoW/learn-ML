# Genetic Algorithm
## step 1 import necessary libraries
```
import copy
import random
import matplotlib.pyplot as plt
import numpy as np
```
## step 2 create a matrix that suggest cities' coordinate
```
# 染色体编码方式 假设25个城市
CITIES = 25
# 生成城市坐标矩阵CITIES*2
cities_axis = np.array([
    [1549, 3408],
    [1561, 2641],
    [3904, 3453],
    [2216, 1177],
    [1563, 4906],
    [3554, 827],
    [2578, 4370],
    [3358, 2054],
    [143, 4789],
    [610, 774],
    [1557, 4064],
    [771, 1823],
    [4753, 4192],
    [2037, 1897],
    [4692, 1887],
    [839, 415],
    [4314, 2696],
    [428, 3626],
    [2725, 543],
    [2349, 263],
    [770, 2550],
    [1627, 1361],
    [2139, 3908],
    [1977, 2775],
    [4345, 11]
])
```
## step 3 define some functions
```
# 基因突变函数，选择两个位点，将位点之间的基因逆序原位置处逆序排列
def gene_mutate(population):
    for piece in population[1:]:
        a = random.random()
        if a <= 0.80:
            index1, index2 = sorted(random.sample(range(1, CITIES), 2))
            piece[index1:index2] = piece[index1:index2][::-1]
            piece[CITIES] = -1


# 定义交配函数，随机选择两个优秀父本，生成子代
def cross_mate(population, num_population):
    parent_1, parent_2 = sorted(random.sample(range(1, int(num_population / 2)), 2))
    start, end = sorted(random.sample(range(CITIES), 2))
    child1, child2 = copy.deepcopy(population[parent_1]), copy.deepcopy(population[parent_2])
    child1[start:end], child2[start:end] = child2[start:end], child1[start:end]
    # 使用部分匹配交叉方法
    dict_ = {}
    for i in range(start,end):
        if child1[i] in dict_.values():
            for j in dict_.keys():
                if dict_[j] == child1[i]:
                    dict_[j] = child2[i]
                    break
        else:
            dict_[child1[i]] = child2[i]
    for i in range(start):
        if child1[i] in dict_.keys():
            child1[i] = dict_[child1[i]]
        elif child1[i] in dict_.values():
            for j in dict_.keys():
                if dict_[j] == child1[i]:
                    child1[i] = j
                    break
    for i in range(end, CITIES):
        if child1[i] in dict_.keys():
            child1[i] = dict_[child1[i]]
        elif child1[i] in dict_.values():
            for j in dict_.keys():
                if dict_[j] == child1[i]:
                    child1[i] = j
                    break
    for i in range(start):
        if child2[i] in dict_.keys():
            child2[i] = dict_[child2[i]]
        elif child2[i] in dict_.values():
            for j in dict_.keys():
                if dict_[j] == child2[i]:
                    child2[i] = j
                    break
    for i in range(end, CITIES):
        if child2[i] in dict_.keys():
            child2[i] = dict_[child2[i]]
        elif child2[i] in dict_.values():
            for j in dict_.keys():
                if dict_[j] == child2[i]:
                    child2[i] = j
                    break
```
## step 4 intialize and run
```
# 定义种群数量以及迭代次数以及最短路径记忆矩阵
num_population = 100
num_iteration = 2000
fitness_history = []

# 初始化种群
population_out = []
for i in range(num_population):
    r = list(range(25))
    random.shuffle(r)
    r.extend([-1])
    population_out.append(r)
# 进行遗传算法
for i in range(num_iteration):
    gene_mutate(population_out)
    cross_mate(population_out, num_population)
    for j in population_out:
        if j[CITIES] == -1:
            sum = 0
            for k in range(CITIES - 1):
                sum += np.linalg.norm(cities_axis[j[k + 1]] - cities_axis[j[k]])
            sum += np.linalg.norm(cities_axis[CITIES - 1] - cities_axis[0])
            j[CITIES] = sum
    population_out = sorted(population_out, key=lambda x: x[CITIES])
    population_out = population_out[:num_population - 1]
    population_out.append(population_out[0][:])
    fitness_history.append(population_out[0][CITIES])
print(population_out[0])
plt.plot(range(num_iteration), fitness_history, color='b')
plt.show()
```
