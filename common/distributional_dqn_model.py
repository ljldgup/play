import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

from common.agent import CommonAgent, train_model_1by1
from common.value_based_models import DoubleDQN
from kof.kof_command_mame import get_action_num

data_dir = os.getcwd()


# 实际运算时发现使用sumtree，对使用交叉熵的损失（包括PPO），收敛极慢，原因不明。
# 而使用所有数据则相对收敛的快一些,所以不适用sumtree
# 这个模型训练的非常慢
class DistributionalDQN(DoubleDQN):
    def __init__(self, role, action_num, functions, model_type='distributionalDQN'):
        # 这里的N必须是能得到有限小数分割区间大小的值，不然后面概率重置会出错
        self.N = 41
        super().__init__(role=role, action_num=action_num, functions=functions,
                         model_type=model_type)
        # reward分布的值，用来乘以网络输出，得到reward期望
        self.vmax = 2
        self.vmin = -2
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
        shared_model = self.base_network_build_fn()
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

        model = Model(shared_model.input, probability_output, name=self.model_type)
        model.compile(optimizer=Adam(1e-6), loss='categorical_crossentropy')
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

        # 将multi_steps个步骤的分布前提
        next_reward_distribution = np.roll(current_reward_distribution, -self.multi_steps)

        # 这里shift(-1)把开始移动到之前最后一次
        time = raw_env['time'].reindex(train_index).diff(1).shift(-1).fillna(1).values

        # 注意这种模型输出的是概率，加值要用用分布的rewards_values，而不是之前模型的输出
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

        # 将当前的Q 加上下下一次的各个值，不是模型输出的概率
        reward_value = np.expand_dims(reward, 1) + self.reward_decay * t_reward

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

    # 将分布移到采样点上，输入的是当前reward的值分布和下次reward的概率分布，根据值分布对概率进行调整
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
        CommonAgent.train_model_with_sum_tree(self, folder, round_nums, batch_size, epochs)

    def value_test(self, folder, round_nums):
        # q值分布可视化
        raw_env = self.raw_env_generate(folder, round_nums)
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
        for i in range(self.action_num):
            ax1 = fig1.add_subplot(4, 4, i + 1)
            ax1.hist(train_reward_expection[i], bins=20)
            fig1.legend()


k = 0.05
N = 41
quantile = np.linspace(0, 1, N)
quantile = np.tile(quantile, N).reshape(N, N)


# 分位是预测函数决定的，所以这里quantile应该和y_pred形状一致
# quantile = quantile.swapaxes(1,0)

# 分位数回归损失公式为 u(τ-δ(u<0)),其中τ为分位数即概率，u=z-θ, z为样本，δ(u<0)当u<0取1
# 上式可表达为 样本z大于网络预测值θ，即u>0，则 τu, 样本z小于于网络预测值θ，即u<0,则 (τ-1)u,两边都是负数所以得正
def quantile_regression_loss(y_true, y_pred):
    # 分位数每个分位都要和target中所有分位数进行计算，一次动作需要进行N*N次计算，所以对矩阵进行扩展
    tile_y_true = tf.tile(y_true, [1, 1, N])
    tile_y_true = tf.reshape(tile_y_true, [-1, N, N])
    # 上面reshape后最后一个维度是重复N次的分位数，调换倒数两次的维度，是的最后一维是完整分位数分布，用于和预测值进行计算
    # 理论上调换哪一边都一样
    tile_y_true = tf.transpose(tile_y_true, perm=[0, 2, 1])
    tile_y_pred = tf.tile(y_pred, [1, 1, N])
    tile_y_pred = tf.reshape(tile_y_pred, [-1, N, N])

    u = tile_y_true - tile_y_pred
    u_quantile = tf.zeros_like(u)
    u_quantile += quantile
    # δ(u<0), 注意这里返回的是去绝对值的结果
    # 样本大于等于预测值取u_quantile,否则1 - u_quantile
    cond_quantile = tf.where(u < 0, x=1 - u_quantile, y=u_quantile)
    # 用来去掉不相关的动作
    sign = tf.where(tile_y_true == 0., 0., 1.)
    # quantile 的维度和，y_pred最后2维相同，可以传播
    # 这里加入 论文中平滑曲线， 在原点有一定平滑作用，当|u| <= k/2 取左边，反之右边
    u_abs = tf.abs(u)
    # 这里应该是minimum 不是maximu, 显然用 u_abs * u_abs随着u增大更大，但靠近0才需要这个
    # k使得直线部分上移了一段，所以也经过原点的1/2*u^2能与直线平滑连接
    lk = tf.where(u_abs < k, 0.5 * u_abs * u_abs, k * (u_abs - 0.5 * k))

    loss = cond_quantile * lk * sign
    return tf.reduce_mean(loss)


# 分为数回归没概率投影操作，相对比较容易实现
class QuantileRegressionDQN(DoubleDQN):
    def __init__(self, role, action_num, functions, model_type='QuantileRegressionDQN'):
        super().__init__(role=role, action_num=action_num, functions=functions,
                         model_type=model_type)
        self.train_reward_generate = self.quantile_regression_dqn_train_data

    def choose_action(self, raw_data, action, random_choose=False):
        if random_choose or random.random() > self.e_greedy:
            return random.randint(0, self.action_num - 1)
        else:
            ans = self.predict_model.predict(self.raw_env_data_to_input(raw_data, action))

            # 这里分位数每个回报值概率是均布的，直接求和应该就行
            return ans.sum(axis=2).argmax()

    def build_model(self):
        shared_model = self.base_network_build_fn()
        t_status = shared_model.output
        probability_distribution_layers = []
        t_status = layers.BatchNormalization()(t_status)
        for a in range(self.action_num):
            t_layer = layers.Dense(128, kernel_initializer='he_uniform')(t_status)
            t_layer = layers.LeakyReLU(0.05)(t_layer)
            # 这里输出分位数
            t_layer = layers.Dense(N)(t_layer)
            probability_distribution_layers.append(t_layer)
        probability_output = concatenate(probability_distribution_layers, axis=1)
        probability_output = layers.Reshape((self.action_num, N))(probability_output)

        model = Model(shared_model.input, probability_output, name=self.model_type)
        model.compile(optimizer=Adam(self.lr), loss=quantile_regression_loss)
        # model.compile(optimizer=Adam(lr=0.00001), loss='mse')

        return model

    def quantile_regression_dqn_train_data(self, raw_env, train_env, train_index):
        reward = raw_env['reward'].reindex(train_index)
        action = raw_env['action'].reindex(train_index)
        action = action.astype('int')

        predict_quantile = self.target_model.predict(train_env)

        time = raw_env['time'].reindex(train_index).diff(1).shift(-1).fillna(1).values

        next_reward = reward.copy()
        for i in range(self.multi_steps - 1):
            next_reward = np.roll(next_reward, -1)
            next_reward[time > 0] = 0
            reward += next_reward * self.reward_decay ** (i + 1)

        next_predict_quantile = predict_quantile[range(len(action)), action, :]
        for i in range(self.multi_steps):
            next_predict_quantile = np.roll(next_predict_quantile, -1)
            # 这里是分位数，不再是分布，直接设成0就可以，
            # 加上reward后就一个值，说明只有一个值，与multi steps也吻合
            next_predict_quantile[time > 0] = 0
        quantile_reward = np.zeros_like(predict_quantile)
        quantile_reward[range(len(action)), action, :] = np.expand_dims(reward,
                                                                        1) + next_predict_quantile * self.reward_decay ** self.multi_steps
        td_error = quantile_reward[range(len(action)), action, :] - predict_quantile[range(len(action)), action, :]
        # 先求和分位数
        td_error = abs(td_error).sum(axis=-1)
        return [quantile_reward, td_error, [None, action]]

    def train_model(self, folder, round_nums=[], batch_size=64, epochs=30):
        # KofAgent.train_model(self, folder, round_nums, batch_size, epochs)
        CommonAgent.train_model_with_sum_tree(self, folder, round_nums, batch_size, epochs)

    def value_test(self, folder, round_nums):
        # q值分布可视化
        raw_env = self.raw_env_generate(folder, round_nums)
        train_env, train_index = self.train_env_generate(raw_env)
        t = model.predict_model.predict([env for env in train_env])
        train_reward_expection = np.sum(t, axis=2)
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
        for i in range(self.action_num):
            ax1 = fig1.add_subplot(4, 4, i + 1)
            ax1.hist(train_reward_expection[i], bins=20)
            fig1.legend()


if __name__ == '__main__':
    role = 'iori'
    # 分位数模型目前还有问题，训练后输出的分位数到几千，不知道错误在哪
    model = DistributionalDQN(role, get_action_num(role))
    # model = QuantileRegressionDQN(role, get_action_num(role))
    # model.value_test(1, [1])

    # model.train_model(2, [1])
    # model.multi_steps = 8
    train_model_1by1(model, range(2, 5), range(1, 10))
    '''
    # for i in range(7, 8):
    model.value_test(3, [1])
    model.save_model()

    model.train_model(3, [1], epochs=30)
    model.save_model()

    raw_env = model.raw_env_generate(1, [1])
    train_env, train_index = model.train_env_generate(raw_env)
    train_distribution, td_error, n_action = model.train_reward_generate(raw_env, train_env, train_index)
    t = model.predict_model.predict([np.expand_dims(env[100], 0) for env in train_env])
    '''
