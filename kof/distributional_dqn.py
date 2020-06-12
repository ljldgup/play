import os
import random
import traceback

import numpy as np
from tensorflow.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.optimizers import Adam
from matplotlib import pyplot as plt

from kof.kof_agent import KofAgent, train_model_1by1
from kof.value_based_models import DoubleDQN

data_dir = os.getcwd()


# 实际运算时发现使用sumtree，对使用交叉熵的损失（包括PPO），收敛极慢，原因不明。
# 而使用所有数据则相对收敛的快一些,所以不适用sumtree
# 这个模型训练的非常慢
class DistributionalDQN(DoubleDQN):
    def __init__(self, role, reward_decay=0.94):
        # 这里的N必须是能得到有限小数分割区间大小的值，不然后面概率重置会出错
        self.N = 41
        super().__init__(role=role, model_name='distributional', reward_decay=reward_decay)

        # reward分布的值，用来乘以网络输出，得到reward期望
        self.vmax = 2
        self.vmin = -2
        self.rewards_values = np.linspace(self.vmin, self.vmax, self.N)
        self.rewards_distribution = np.array([[self.rewards_values] * self.action_num])
        self.train_reward_generate = self.distributional_dqn_train_data
        self.copy_interval = 2

    def choose_action(self, raw_data, action, random_choose=False):
        if random_choose or random.random() > self.e_greedy:
            return random.randint(0, self.action_num - 1)
        else:
            ans = self.predict_model.predict(self.raw_env_data_to_input(raw_data, action))

            # 分布乘以对应reward得到期望，然后返回reward期望最大的动作
            return (ans * self.rewards_distribution).sum(axis=2).argmax()

    def build_model(self):
        shared_model = self.build_shared_model()
        t_status = shared_model.output
        probability_distribution_layers = []
        for a in range(self.action_num):
            t_layer = layers.Dense(128, kernel_initializer='he_uniform')(t_status)
            t_layer = layers.LeakyReLU(0.05)(t_layer)
            # 注意这里softmax 不用he_uniform初始化
            t_layer = layers.Dense(self.N, activation='softmax',
                                   name='action_{}_distribution'.format(a))(t_layer)
            probability_distribution_layers.append(t_layer)
        probability_output = concatenate(probability_distribution_layers, axis=1)
        probability_output = layers.Reshape((self.action_num, self.N))(probability_output)

        model = Model(shared_model.input, probability_output, name=self.model_name)
        model.compile(optimizer=Adam(), loss='categorical_crossentropy')
        # model.compile(optimizer=Adam(lr=0.00001), loss='mse')

        return model

    def distributional_dqn_train_data(self, raw_env, train_env, train_index):
        reward = raw_env['reward'].reindex(train_index)
        action = raw_env['action'].reindex(train_index)
        action = action.astype('int')

        target_model_prediction = self.target_model.predict(train_env)
        predict_model_prediction = self.predict_model.predict(train_env)

        # 因为是分布所以要先乘以值获得期望求最大值
        pre_actions = (predict_model_prediction * self.rewards_distribution).sum(axis=2).argmax(axis=1)

        next_max_reward_action = (predict_model_prediction * self.rewards_distribution).sum(axis=2).argmax(axis=1)
        current_reward_distribution = target_model_prediction[
            range(len(train_index)), next_max_reward_action.astype('int')]

        next_reward_distribution = np.roll(current_reward_distribution, -self.multi_steps)

        # 这里shift(-1)把开始移动到之前最后一次
        time = raw_env['time'].reindex(train_index).diff(1).shift(-1).fillna(1).values

        t_reward = np.array([self.rewards_values] * len(train_index))
        for i in range(self.multi_steps):
            np.roll(t_reward, -1)
            # 这里让累积到最后一次的所有下次值分布为0，那么相加后返回分布值就都是r
            # 分配概率时就只按r值分配，概率本身无影响，next_reward_distribution不需要再操作什么
            t_reward[time > 0] = 0

        next_reward = reward.copy()
        for i in range(self.multi_steps - 1):
            next_reward = np.roll(next_reward, -1)
            # 从下一个round往上移动的，reward为0
            next_reward[time > 0] = 0
            reward += next_reward * self.reward_decay ** (i + 1)

        # 将当前的Q 加上下下一次的各个值，取得下一次可能得到得值，对应model输出的概率
        reward_value = np.expand_dims(reward.values, 1) + self.reward_decay * t_reward

        # 裁剪
        reward_value[reward_value > self.vmax] = self.vmax
        reward_value[reward_value < self.vmin] = self.vmin

        # 配合预测的概率进行重新组织概率，使值和均匀分布
        new_distribution = self.gen_new_reward_distribution(reward_value, next_reward_distribution)

        # 改用交叉熵
        td_error = new_distribution * np.log(predict_model_prediction[range(len(train_index)), action])
        td_error = td_error.sum(axis=1)

        predict_model_prediction[range(len(train_index)), action] = new_distribution
        '''
        # 这里因为原来的网络训练不动，可能是其他无关分布限制，这里将其他分布置0，使其在loss中占比为0
        # 效果也很一般
        reward_distribution = np.zeros(shape=predict_model_prediction.shape)
        reward_distribution[range(len(train_index)), action] = new_distribution
        '''
        return [predict_model_prediction, td_error, [pre_actions, action.values]]

    # 将分布移到采样点上
    def gen_new_reward_distribution(self, reward_value, old_reward_distribution):
        length = len(reward_value)
        interval = (self.vmax - self.vmin) / (self.N - 1)
        reward_int = (reward_value / interval).astype('int')
        reward_distance = interval - abs(reward_value - reward_int * interval)
        # 调整对应的位置，后续作为坐标使用
        reward_int += self.N // 2

        '''
        new_distribution = np.zeros(old_reward_distribution.shape)
        for i in range(len(old_reward_distribution)):
            for n in range(self.N):
                t = old_reward_distribution[i][reward_int[i] == n] * reward_distance[i][reward_int[i] == n]
                new_distribution[i][n] += t.sum()
        '''
        # 原来的方法是每行，对每种动作类型的概率求和，相对较慢
        # 这里改成每列操作，对于reward_value中一列，
        # 根据reward_int取出新分布每行中对应列，然后加上旧概率乘以距离除以间距即可
        # 编程的时候尽可能通过索引的操作代替循环
        new_distribution_2 = np.zeros(old_reward_distribution.shape)
        for n in range(self.N):
            each_row_index = (range(length), reward_int[:, n])
            new_distribution_2[each_row_index] += old_reward_distribution[:, n] * reward_distance[:, n] / interval

        reward_int = (reward_value / interval).astype('int')
        reward_int[reward_int < 0] -= 1
        reward_int[reward_int > 0] += 1
        reward_int += self.N // 2
        reward_distance = interval - reward_distance

        '''
        for i in range(len(old_reward_distribution)):
            for n in range(self.N):
                t = old_reward_distribution[i][reward_int[i] == n] * reward_distance[i][reward_int[i] == n]
                new_distribution[i][n] += t.sum()
        '''

        # 避免越界
        reward_int[reward_int >= self.N] = self.N - 1
        reward_int[reward_int < 0] = 0
        for n in range(self.N):
            each_row_index = (range(length), reward_int[:, n])
            new_distribution_2[each_row_index] += old_reward_distribution[:, n] * reward_distance[:, n] / interval

        return new_distribution_2

    def train_model(self, folder, round_nums=[], batch_size=64, epochs=30):
        KofAgent.train_model(self, folder, round_nums, batch_size, epochs)

    def value_test(self, folder, round_nums):
        # q值分布可视化
        raw_env = self.raw_data_generate(folder, round_nums)
        train_env, train_index = self.train_env_generate(raw_env)
        train_reward, td_error, n_action = self.train_reward_generate(raw_env, train_env, train_index)
        train_reward_expection = np.sum(train_reward * self.rewards_values, axis=2)
        # 这里是训练数据但是也可以拿来参考,查看是否过估计，目前所有的模型几乎都会过估计
        print('max reward:', train_reward_expection.max())
        print('min reward:', train_reward_expection.min())
        '''
        # 这里所有的图在一个图上，plt.figure()
        # 这里不压平flatten会按照第一个维度动作数来统计
        plt.hist(train_reward.flatten(), bins=30, label=self.model_name)
        # 加了这句才显示lable
        plt.legend()
        '''

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.hist(train_reward_expection.flatten(), bins=30, label=self.model_name)
        fig1.legend()


if __name__ == '__main__':
    model = DistributionalDQN('iori')
    train_model_1by1(model, range(10), range(1, 7))
    model.value_test(1, [1])
    '''
    raw_env = model.raw_data_generate(1, [1])
    train_env, train_index = model.train_env_generate(raw_env)
    train_distribution, td_error, n_action = model.distributional_dqn_train_data(raw_env, train_env, train_index)
    t = model.predict_model.predict([np.expand_dims(env[100], 0) for env in train_env])
    '''
