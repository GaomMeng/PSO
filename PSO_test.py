# -*- encoding = utf-8 -*-
# @Time : 2022/6/28 17:39
# @Author : gm
# @File : PSO_test.py
# @Software : PyCharm

import numpy as np
import random
import matplotlib.pyplot as plt
import time
import copy
import argparse
import copy

"""==============================     下面是具体算法    ================================"""

"""

PSO基本原理：模仿鸟群觅食

PSO架构：
（1）、Particle类：定义粒子，主要记录粒子的位置、速度、适应值、历史最优位置
（2）、PSO类：算法主体，包含算法细节

"""


class Particle:
    """
    初始化参数表：

    x_max：解的边界，任意x解应该属于区间[-x_max, x_max]
    max_vel: 最大速度
    dim：问题维数，即解的维数
    config：初始化列表，主要包含一个具有适应度函数的类，可根据需要传入其他参数，例如：   config = {'plm': plm, key1: value1, key2: value2, ...}

    plm:定义的一个类，根据具体情况修改。例如设适应值为二次函数 y=100 - x^2 ， 定义如下：

    class Function:
        def __init__(self):
            pass

        def test(self, test_config):
            x = test_config['x']
            y = 100 - x^2
            return y
    """

    # 初始化
    def __init__(self, x_max, max_vel, dim, config):
        # np.random.seed(int(str(time.time() % 10)[-6:]))
        # time.sleep(0.001)
        self.__pos = list(np.random.uniform(-x_max, x_max, dim))  # 粒子的位置
        self.__vel = list(np.random.uniform(-max_vel, max_vel, dim))  # 粒子的速度

        self.__bestPos = [0.0 for _ in range(dim)]  # 粒子最好的位置
        self.config = config
        self.config['x'] = self.__pos

        # self.plm 是一个包含适应度函数的一个类，该类包含了一个test()函数，将解输入至test()函数，即可获取适应值
        self.plm = self.config['plm']
        self.__fitnessValue = self.plm.test(self.config)  # 适应度函数值

    def set_pos(self, i, value):
        self.__pos[i] = value

    def get_pos(self):
        return self.__pos

    def set_best_pos(self, i, value):
        self.__bestPos[i] = value

    def get_best_pos(self):
        return self.__bestPos

    def set_vel(self, i, value):
        self.__vel[i] = value

    def get_vel(self):
        return self.__vel

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue


class PSO:
    """
    初始化参数表：

    args：参数列表，包含所需要的所有参数，例如迭代次数、种群边界等等
    config：初始化列表，主要包含一个具有适应度函数的类，可根据需要传入其他参数，例如：   config = {'plm': plm, key1: value1, key2: value2, ...}

    plm:定义的一个类，根据具体情况修改。例如设适应值为二次函数 y=100 - x^2 ， 定义如下：

    class Function:
        def __init__(self):
            pass

        def test(self, test_config):
            x = test_config['x']
            y = 100 - x^2
            return y

    关键函数：
    update()  运行update即可得到粒子群优化结果，详细见具体函数

    """

    def __init__(self, args, config):
        self.config = config
        self.C1 = args.Individual_learning_factor
        self.C2 = args.Social_learning_factor
        self.W = args.Inertia_weight
        self.dim = args.x_dim  # 粒子的维度
        self.size = args.Population_size  # 粒子个数
        self.iter_num = args.Iteration_number  # 迭代次数
        self.x_max = args.x_bound
        self.max_vel = args.Max_vel  # 粒子最大速度
        self.best_fitness_value = self.config[
            'best_fitness_value'] if 'best_fitness_value' in self.config.keys() else float('-Inf')
        self.best_position = [0.0 for i in range(self.dim)]  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.plm = self.config['plm']

        # 对种群进行初始化
        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim, self.config) for i in range(self.size)]

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, i, value):
        self.best_position[i] = value

    def get_bestPosition(self):
        return self.best_position

    # 更新速度
    def update_vel(self, part):
        for i in range(self.dim):
            vel_value = self.W * part.get_vel()[i] + self.C1 * random.random() * (
                        part.get_best_pos()[i] - part.get_pos()[i]) \
                        + self.C2 * random.random() * (self.get_bestPosition()[i] - part.get_pos()[i])
            if vel_value > self.max_vel:
                vel_value = self.max_vel
            elif vel_value < -self.max_vel:
                vel_value = -self.max_vel
            part.set_vel(i, vel_value)

    # 更新位置
    def update_pos(self, part):
        for i in range(self.dim):
            pos_value = part.get_pos()[i] + part.get_vel()[i]
            pos_value = np.clip([copy.deepcopy(pos_value)], -5, 5)[0]
            part.set_pos(i, pos_value)
        self.config['x'] = part.get_pos()
        value = self.plm.test(self.config)
        if value > part.get_fitness_value():
            part.set_fitness_value(value)
            for i in range(self.dim):
                part.set_best_pos(i, part.get_pos()[i])
        if value > self.get_bestFitnessValue():
            self.set_bestFitnessValue(value)
            for i in range(self.dim):
                self.set_bestPosition(i, part.get_pos()[i])

    def update(self):
        """
        返回值为一个字典，包含你需要的所有信息，可根据具体情况增改
        """
        pop = []
        x_list = []
        fit_list = []
        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)  # 更新速度
                self.update_pos(part)  # 更新位置
                pop.append([copy.deepcopy(part.get_pos()), copy.deepcopy(part.get_fitness_value())])
                x_list.append(copy.deepcopy(part.get_pos()))
                fit_list.append(copy.deepcopy(part.get_fitness_value()))

        return_dict = {'pop': pop,
                       'x_list': x_list,
                       'fit_list': fit_list,
                       'best_x': self.get_bestPosition(),
                       'best_v': self.get_bestFitnessValue()}
        return return_dict


"""==============================     下面是一个测试案例    ================================"""

# 定义一个测试问题 y = 100 - x * x
class test_function:
    def __init__(self):
        pass

    def test(self, config):
        x = config['x']
        y = 100 - np.dot(x, x)
        return y


# 定义所有需要的参数
def set_param():
    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', default=10, type=int)
    parser.add_argument('--x_bound', default=5., type=float)
    parser.add_argument('--Population_size', default=100, type=int)
    parser.add_argument('--Iteration_number', default=100, type=int)
    parser.add_argument('--Inertia_weight', default=0.75, type=float)
    parser.add_argument('--Individual_learning_factor', default=1.4, type=float)
    parser.add_argument('--Social_learning_factor', default=1.4, type=float)
    parser.add_argument('--Max_vel', default=0.8, type=float)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_param()
    test_problem = test_function()
    pso_config = {'name': 'test', 'plm': test_problem}
    pso = PSO(args, pso_config)
    pso_dict = pso.update()

    print(f"   the max value is : {pso_dict['best_v']}    the best position is {pso_dict['best_x']}   ")










