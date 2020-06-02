import os
import random
import traceback

import numpy as np

from kof.kof_agent import KofAgent
from tensorflow.keras import layers
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import concatenate, BatchNormalization, CuDNNLSTM
from tensorflow.python.keras.optimizers import Adam

data_dir = os.getcwd()


class DistributionalDQN(KofAgent):

    def __init__(self, role, reward_decay=0.96):

        self.N = 21
        super().__init__(role=role, model_name='distributional', reward_decay=reward_decay)

        # reward分布的值，用来乘以网络输出，得到reward期望
        self.vmax = 1
        self.vmin = -1
        self.rewards_values = np.linspace(self.vmin, self.vmax, self.N)
        self.rewards_distribution = np.array([[self.rewards_values] * self.action_num])
        self.train_reward_generate = self.distributional_dqn_train_data

    def choose_action(self, raw_data, action, random_choose=False):
        if random_choose or random.random() > self.e_greedy:
            return random.randint(0, self.action_num - 1)
        else:
            ans = self.predict_model.predict(self.raw_env_data_to_input(raw_data, action))

            # 分布乘以对应reward得到期望，然后返回reward期望最大的动作
            return (ans * self.rewards_distribution).sum(axis=2).argmax()

    def build_model(self):
        role1_actions = Input(shape=(self.input_steps,), name='role1_actions')
        role2_actions = Input(shape=(self.input_steps,), name='role2_actions')
        role1_actions_embedding = layers.Embedding(512, 8, name='role1_actions_embedding')(role1_actions)
        role2_actions_embedding = layers.Embedding(512, 8, name='role2_actions_embedding')(role2_actions)

        role1_energy = Input(shape=(self.input_steps,), name='role1_energy')
        role1_energy_embedding = layers.Embedding(5, 2, name='role1_energy_embedding')(role1_energy)
        role2_energy = Input(shape=(self.input_steps,), name='role2_energy')
        role2_energy_embedding = layers.Embedding(5, 2, name='role2_energy_embedding')(role2_energy)

        # 爆气状态
        role2_baoqi = Input(shape=(self.input_steps,), name='role2_baoqi')
        role2_baoqi_embedding = layers.Embedding(2, 2, name='role2_baoqi_embedding')(role2_baoqi)

        role1_x_y = Input(shape=(self.input_steps, 2), name='role1_x_y')
        role2_x_y = Input(shape=(self.input_steps, 2), name='role2_x_y')

        role_position = concatenate([role1_x_y, role2_x_y])
        conv_position = layers.SeparableConv1D(8, 1, padding='same', strides=1, kernel_initializer='he_uniform')(
            role_position)
        conv_position = BatchNormalization()(conv_position)
        conv_position = layers.LeakyReLU(0.05)(conv_position)

        role_distance = layers.Subtract()([role1_x_y, role2_x_y])
        role_distance = BatchNormalization()(role_distance)

        concatenate_status = concatenate(
            [role1_actions_embedding, role_distance, conv_position, role1_energy_embedding,
             role2_actions_embedding, role2_energy_embedding, role2_baoqi_embedding])

        lstm_status = CuDNNLSTM(512)(concatenate_status)
        # bn层通常加在线性输出(cnn,fc)后面，应为线性输出分布均衡，加在rnn后面效果很差
        # t_status = BatchNormalization()(lstm_status)
        # 加了一层fc效果好了很多，直接加载rnn上训练效果很一般
        t_status = layers.Dense(512, kernel_initializer='he_uniform')(lstm_status)
        t_status = BatchNormalization()(t_status)
        t_status = layers.LeakyReLU(0.05)(t_status)
        probability_distribution_layers = []
        for a in range(self.action_num):
            t_layer = layers.Dense(self.N, kernel_initializer='he_uniform',
                                   name='action_{}_distribution'.format(a))(t_status)
            # 注意这里softmax 不用he_uniform初始化
            t_layer = layers.Softmax()(t_layer)
            probability_distribution_layers.append(t_layer)
        probability_output = concatenate(probability_distribution_layers, axis=1)
        probability_output = layers.Reshape((self.action_num, self.N))(probability_output)

        model = Model(
            [role1_actions, role2_actions, role1_energy, role2_energy, role1_x_y, role2_x_y,
             role2_baoqi],
            probability_output)

        model.compile(optimizer=Adam(lr=0.00002), loss='categorical_crossentropy')

        return model

    def raw_env_data_to_input(self, raw_data, action):
        return [raw_data[:, :, 0], raw_data[:, :, 1], raw_data[:, :, 2], raw_data[:, :, 3],
                raw_data[:, :, 4:6], raw_data[:, :, 6:8], raw_data[:, :, 8]]

    def empty_env(self):
        return [[], [], [], [], [], [], []]

    def distributional_dqn_train_data(self, raw_env, train_env, train_index):
        reward = raw_env['reward'].reindex(train_index)
        action = raw_env['action'].reindex(train_index)
        action = action.astype('int')

        target_model_prediction = self.target_model.predict(train_env)
        predict_model_prediction = self.predict_model.predict(train_env)

        # 因为是分布所以要先乘以值获得期望求最大值
        pre_actions = (predict_model_prediction * self.rewards_distribution).sum(axis=2).argmax(axis=1)

        next_max_reward_action = (predict_model_prediction * self.rewards_distribution).sum(axis=2).argmax(axis=1)
        next_reward_distribution = target_model_prediction[
            range(len(train_index)), next_max_reward_action.astype('int')]
        current_reward_distribution = next_reward_distribution
        next_reward_distribution = np.roll(next_reward_distribution, -1)

        # 这里shift(-1)把开始移动到之前最后一次
        time = raw_env['time'].reindex(train_index).diff(1).shift(-1).fillna(1).values
        # 最后一次仍然用之前的分布
        next_reward_distribution[time > 0] = current_reward_distribution[time > 0]

        t_reward = np.array([self.rewards_values] * len(train_index))
        reward_value = np.expand_dims(reward.values, 1) + self.reward_decay * t_reward

        # 裁剪
        reward_value[reward_value > self.vmax] = self.vmax
        reward_value[reward_value < self.vmin] = self.vmin
        new_distribution = self.gen_new_reward_distribution(reward_value, next_reward_distribution)
        td_error = (new_distribution - predict_model_prediction[range(len(train_index)), action]) * \
                   self.rewards_distribution[0, 0]
        td_error = td_error.sum(axis=1)
        predict_model_prediction[range(len(train_index)), action] = new_distribution

        return [predict_model_prediction, td_error, [pre_actions, action.values]]

    # 将分布移到采样点上
    def gen_new_reward_distribution(self, reward_value, old_reward_distribution):
        length = len(reward_value)
        reward_int = (reward_value / 0.1).astype('int')
        reward_distance = 1 - 10 * abs(reward_value - reward_int * 0.1)
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
        # 这里改成每列操作，对于reward_value中一列，根据reward_int取出新分布每行中对应列，然后加上旧概率乘以距离即可
        # 一部分循环通过索引，交给numpy,效率有提升
        # 编程的时候尽可能通过索引的操作代替循环
        new_distribution_2 = np.zeros(old_reward_distribution.shape)
        for n in range(self.N):
            each_row_index = (range(length), reward_int[:, n])
            new_distribution_2[each_row_index] += old_reward_distribution[:, n] * reward_distance[:, n]

        reward_int = (reward_value / 0.1).astype('int')
        reward_int[reward_int < 0] -= 1
        reward_int[reward_int > 0] += 1
        reward_int += self.N // 2
        reward_distance = 1 - reward_distance

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
            new_distribution_2[each_row_index] += old_reward_distribution[:, n] * reward_distance[:, n]

        return new_distribution_2


if __name__ == '__main__':
    model = DistributionalDQN('iori')

    for i in range(11, 14):
        try:
            print('train ', i)
            for num in range(1, 10):
                model.train_model_with_sum_tree(i, [num], epochs=30)
                model.weight_copy()
        except:
            # print('no data in ', i)
            traceback.print_exc()
        model.save_model()
    '''
    raw_env = model.raw_data_generate(1, [1])
    train_env, train_index = model.train_env_generate(raw_env)
    train_distribution, n_action = model.distributional_dqn_train_data(raw_env, train_env, train_index)
    t = model.predict_model.predict([np.expand_dims(env[100], 0) for env in train_env])
    '''
