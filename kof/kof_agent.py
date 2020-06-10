import os
import random
import time

import pandas as pd
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import concatenate, BatchNormalization, CuDNNLSTM
from kof.kof_command_mame import global_set, role_commands
from kof.sumtree import SumTree


class RandomAgent:
    def __init__(self, role):
        self.model_name = 'random'
        self.role = role
        self.action_num = len(role_commands[role])
        self.input_steps = 1
        global_set(role)

    def choose_action(self, *args, **kwargs):
        return random.randint(0, self.action_num - 1)

    def save_model(self):
        pass


data_dir = os.getcwd()
env_colomn = ['role1_action', 'role2_action',
              'role1_energy', 'role2_energy',
              'role1_position_x', 'role1_position_y',
              'role2_position_x', 'role2_position_y', 'role1_baoqi', 'role2_baoqi', 'role1', 'role2',
              'role1_guard_value',
              'role1_combo_count',
              'role1_life', 'role2_life',
              'time', 'coins', ]


# 选择效果比较好，即文件比较大的记录
def get_maxsize_file(folder):
    files = os.listdir('{}/{}'.format(data_dir, folder))
    data_files = filter(lambda f: '.' in f, files)
    round_nums = list(set([file.split('.')[0] for file in data_files]))
    round_nums.sort()
    file_size = map(lambda num: os.path.getsize('{}/{}/{}.env'.format(data_dir, folder, num), round_nums))
    # 取前6个
    return np.array(file_size).argsort()[0:6]


class KofAgent:
    def __init__(self, role, model_name,
                 reward_decay=0.94,
                 input_steps=8):
        self.role = role
        self.model_name = model_name
        self.reward_decay = reward_decay
        self.e_greedy = 0.95
        # 输入步数
        self.input_steps = input_steps
        self.multi_steps = 1
        global_set(role)
        self.action_num = len(role_commands[role])
        self.predict_model = self.build_model()
        self.record = self.get_record()
        # 模型参数拷贝间隔
        self.copy_interval = 6
        if os.path.exists('{}/model/{}_{}.index'.format(data_dir, self.role, self.model_name)):
            print('load model {}'.format(self.role))
            self.load_model()

    def choose_action(self, raw_data, action, random_choose=False):
        if random_choose or random.random() > self.e_greedy:
            return random.randint(0, self.action_num - 1)
        else:
            # 根据dqn的理论，这里应该用训练的模型， target只在寻训练的时候使用，并且一定时间后才复制同步
            return self.predict_model.predict(self.raw_env_data_to_input(raw_data, action)).argmax()

    # 读入round_nums文件下的文件，并计算reward
    def raw_data_generate(self, folder, round_nums):
        if round_nums:
            if os.path.exists('{}/{}/{}.env'.format(data_dir, folder, round_nums[0])):
                raw_env = np.loadtxt('{}/{}/{}.env'.format(data_dir, folder, round_nums[0]))
                raw_actions = np.loadtxt('{}/{}/{}.act'.format(data_dir, folder, round_nums[0]))
            else:
                raise Exception('no {}/{}/{}.env'.format(data_dir, folder, round_nums[0]))
            for i in range(1, len(round_nums)):
                if os.path.exists('{}/{}/{}.env'.format(data_dir, folder, round_nums[i])):
                    # print(i)
                    raw_env = np.concatenate([raw_env,
                                              np.loadtxt('{}/{}/{}.env'.format(data_dir, folder, round_nums[i]))])
                    raw_actions = np.concatenate([raw_actions,
                                                  np.loadtxt(
                                                      '{}/{}/{}.act'.format(data_dir, folder, round_nums[i]))])

            raw_env = pd.DataFrame(raw_env, columns=env_colomn)
            raw_env['action'] = raw_actions
            # 筛选时间在流动的列
            raw_env = raw_env[raw_env['time'].diff(1).fillna(0) != 0]
            life_reward = raw_env['role1_life'].diff(1).fillna(0) - raw_env['role2_life'].diff(1).fillna(0)
            # 这里避免新的一局开始，血量回满被误认为报酬
            life_reward[raw_env['time'].diff(1) > 0] = 0

            # 防守的收益
            # guard_reward = -raw_env['guard_value'].diff(1).fillna(0)
            # guard_reward = guard_reward.map(lambda x: x if x > 0 else 0)

            # 生成time_steps时间步内的reward
            # 改成dqn 因为自动加后面一次报酬，后应该不需要rolling
            # reward_sum = reward.rolling(self.reward_steps, min_periods=1).sum().shift(-self.reward_steps).fillna(0)

            # 值要在[-1，1]左右,reward_sum太小反而容易过估计
            raw_env['raw_reward'] = life_reward
            raw_env['reward'] = life_reward / 40

            # 当前步的reward实际上是上一步的，我一直没有上移，这是个巨大的错误
            raw_env['reward'] = raw_env['reward'].shift(-1).fillna(0)

            # 根据胜负增加额外的报酬,pandas不允许切片或者搜索赋值，只能先这样
            end_index = (raw_env[raw_env['time'].diff(1).shift(-1).fillna(1) > 0]).index
            for idx in end_index:
                # 这里暂时不明白为什么是loc,我是按索引取得，按理应该是iloc
                if raw_env.loc[idx]['role1_life'] > raw_env.loc[idx]['role2_life']:
                    raw_env.loc[idx]['reward'] = 0.5
                else:
                    raw_env.loc[idx]['reward'] = -0.5

            # 使用log(n+x)-log(n)缩放reward，防止少量特别大的动作影响收敛，目前来看适当的缩放，收敛效果好。
            # raw_env['reward'] = reward.map(
            #     lambda x: math.log10(100 + x) - math.log10(100) if x > 0 else -math.log10(100 - x) + math.log10(100))

            return raw_env

    # 生成未加衰减奖励的训练数据
    # 这里把文件夹换成env_reward_generate生成的数据，避免过度耦合
    def train_env_generate(self, raw_env):
        train_env_data = self.empty_env()
        train_index = []

        # 直接打乱数据
        # index_list = raw_env[raw_env['reward'] != 0].index.to_list()
        for index in raw_env['reward'].index:
            # 这里是loc取的数量是闭区间和python list不一样
            # guard_value不用，放在这里先补齐
            env = raw_env[['role1_action', 'role2_action',
                           'role1_energy', 'role2_energy',
                           'role1_position_x', 'role1_position_y',
                           'role2_position_x', 'role2_position_y', 'role1_baoqi', 'role2_baoqi']].loc[
                  index - self.input_steps + 1:index]
            action = raw_env['action'].loc[index - self.input_steps:index - 1]

            # 之前能够去除time_steps个连续数据，在操作
            # 这里考虑到输入与步长，所以加大了判断长度 3 * self.input_steps
            if len(action) != self.input_steps or env.index[-1] - env.index[0] > self.input_steps - 1:
                pass
            else:
                data = env.values
                data = data.reshape(1, *data.shape)
                action = action.values
                action = action.reshape(1, *action.shape)
                split_data = self.raw_env_data_to_input(data, action)

                for i in range(len(split_data)):
                    train_env_data[i].append(split_data[i])

                train_index.append(index)

        train_env_data = [np.array(env) for env in train_env_data]
        # 去掉原来的长度为1的sample数量值
        train_env_data = [env.reshape(env.shape[0], *env.shape[2:]) for env in train_env_data]

        return [train_env_data, train_index]

    # 最开始用的不考虑长期收益的模型，在pandas钟使用rolling来使其拥有少量的长期效果
    # 现在使用该函数实现多态
    def train_reward_generate(self, raw_env, train_env, train_index):
        return [None, [None, None]]

    # 在线学习可以batch_size设置成
    # 改成所有数据何在一起，打乱顺序，使用batch 训练，速度快了很多
    # batch_size的选取不同，损失表现完全不一样
    def train_model(self, folder, round_nums=[], batch_size=64, epochs=30):

        raw_env = self.raw_data_generate(folder, round_nums)
        train_env, train_index = self.train_env_generate(raw_env)
        train_target, td_error, action = self.train_reward_generate(raw_env, train_env, train_index)

        random_index = np.random.permutation(len(train_index))
        # verbose参数控制输出，这里每个epochs输出一次
        self.predict_model.fit([env[random_index] for env in train_env], train_target[random_index],
                               batch_size=batch_size,
                               epochs=epochs, verbose=2)

        self.record['total_epochs'] += epochs

    # Prioritized Replay DQN 使用sumtree 生成 batch
    def train_model_with_sum_tree(self, folder, round_nums=[], batch_size=64, epochs=30):
        if not round_nums:
            files = os.listdir('{}/{}'.format(data_dir, folder))
            data_files = filter(lambda f: '.' in f, files)
            round_nums = list(set([file.split('.')[0] for file in data_files]))

        raw_env = self.raw_data_generate(folder, round_nums)
        train_env, train_index = self.train_env_generate(raw_env)
        train_target, td_error, action = self.train_reward_generate(raw_env, train_env, train_index)
        sum_tree = SumTree(abs(td_error))
        loss_history = []
        print('train {}/{} {} epochs'.format(folder, round_nums, epochs))
        for e in range(epochs):
            loss = 0
            for i in range(len(train_target) // batch_size):
                index = sum_tree.gen_batch_index(batch_size)
                loss += self.predict_model.train_on_batch([env[index] for env in train_env], train_target[index])
            loss_history.append(loss)
        for loss in loss_history:
            # 根据标准公式，这里需要求个均值
            print(loss / (len(train_index) // batch_size))
        self.record['total_epochs'] += epochs

    def save_model(self):
        if not os.path.exists('{}/model'.format(data_dir)):
            os.mkdir('{}/model'.format(data_dir))
        self.predict_model.save_weights('{}/model/{}_{}'.format(data_dir, self.role, self.model_name))

        with open('{}/model/{}_{}_record'.format(data_dir, self.role, self.model_name), 'w') as r:
            r.write(str(self.record))

    def load_model(self):
        self.predict_model.load_weights('{}/model/{}_{}'.format(data_dir, self.role, self.model_name))

    # 模型输入+lstm的公用部分
    def build_shared_model(self):
        role1_actions = Input(shape=(self.input_steps,), name='role1_actions')
        role2_actions = Input(shape=(self.input_steps,), name='role2_actions')
        role1_actions_embedding = layers.Embedding(512, 8, name='role1_actions_embedding')(role1_actions)
        role2_actions_embedding = layers.Embedding(512, 8, name='role2_actions_embedding')(role2_actions)

        role1_energy = Input(shape=(1,), name='role1_energy')
        role1_energy_embedding = layers.Embedding(5, 2, name='role1_energy_embedding')(role1_energy)
        role2_energy = Input(shape=(1,), name='role2_energy')
        role2_energy_embedding = layers.Embedding(5, 2, name='role2_energy_embedding')(role2_energy)

        role1_baoqi = Input(shape=(1,), name='role1_baoqi')
        role1_baoqi_embedding = layers.Embedding(2, 1, name='role1_baoqi_embedding')(role1_baoqi)
        role2_baoqi = Input(shape=(1,), name='role2_baoqi')
        role2_baoqi_embedding = layers.Embedding(2, 1, name='role2_baoqi_embedding')(role2_baoqi)

        role1_x_y = Input(shape=(self.input_steps, 2), name='role1_x_y')
        role2_x_y = Input(shape=(self.input_steps, 2), name='role2_x_y')

        role_position = concatenate([role1_x_y, role2_x_y])
        conv_position = layers.SeparableConv1D(8, 1, padding='same', strides=1, kernel_initializer='he_uniform')(
            role_position)
        conv_position = BatchNormalization()(conv_position)
        conv_position = layers.LeakyReLU(0.05)(conv_position)

        role_distance = layers.Subtract()([role1_x_y, role2_x_y])
        conv_distance = layers.SeparableConv1D(4, 2, padding='same', strides=1, kernel_initializer='he_uniform')(
            role_distance)
        conv_distance = BatchNormalization()(conv_distance)

        action_input = Input(shape=(self.input_steps,), name='action_input')
        action_input_embedding = layers.Embedding(self.action_num, 4, name='action_input_embedding')(action_input)

        concatenate_status = concatenate(
            [role1_actions_embedding, conv_position, conv_distance,
             action_input_embedding, role2_actions_embedding])
        lstm_status = CuDNNLSTM(512)(concatenate_status)

        t_status = layers.Dense(512, kernel_initializer='he_uniform')(lstm_status)
        t_status = BatchNormalization()(t_status)
        t_status = layers.LeakyReLU(0.05)(t_status)

        # 曝气应该是个综合性影响，所以直接加在最后
        t_status = layers.concatenate(
            [t_status, K.squeeze(role1_baoqi_embedding, 1), K.squeeze(role2_baoqi_embedding, 1),
             K.squeeze(role1_energy_embedding, 1),
             K.squeeze(role2_energy_embedding, 1)])

        shared_model = Model([role1_actions, role2_actions, role1_energy, role2_energy,
                              role1_x_y, role2_x_y, role1_baoqi, role2_baoqi, action_input], t_status)
        # 这里模型不能编译，不然后面无法扩充
        return shared_model

    # 游戏运行时把原始环境输入，分割成模型能接受的输入，在具体的模型可以修改
    def raw_env_data_to_input(self, raw_data, action):
        # 这里energy改成一个只输入最后一个,这里输出的形状应该就是1，貌似在keras中也能正常运作
        # 动作，空间取所有，状态类的只取最后步
        return [raw_data[:, :, 0], raw_data[:, :, 1], raw_data[:, -1, 2], raw_data[:, -1, 3],
                raw_data[:, :, 4:6], raw_data[:, :, 6:8], raw_data[:, -1, 8], raw_data[:, -1, 9], action]

    # 这里返回的list要和raw_env_data_to_input返回的大小一样
    def empty_env(self):
        return [[], [], [], [], [], [], [], [], []]

    def get_record(self):
        if os.path.exists('{}/model/{}_{}_record'.format(data_dir, self.role, self.model_name)):
            with open('{}/model/{}_{}_record'.format(data_dir, self.role, self.model_name), 'r') as r:
                record = eval(r.read())
        else:
            record = {'total_epochs': 0, 'begin_time': time.asctime(time.localtime(time.time()))}
        return record

    # 分析动作的出现频率和回报率
    def operation_analysis(self, folder):
        files = os.listdir('{}/{}'.format(data_dir, folder))
        data_files = filter(lambda f: '.' in f, files)
        round_nums = list(set([file.split('.')[0] for file in data_files]))

        raw_env = self.raw_data_generate(folder, [round_nums[0]])
        raw_env['num'] = round_nums[0]
        for i in range(1, len(round_nums)):
            t = self.raw_data_generate(folder, [round_nums[i]])
            t['num'] = round_nums[i]
            raw_env = pd.concat([raw_env, t])

        raw_env['num'] = raw_env['num'].astype('int')
        raw_env['action'] = raw_env['action'].astype('int')

        # 注意这里对num，action进行聚类，之后他们都在层次化索引上
        # 所以需要unstack，将action移到列名,才能绘制出按文件名分开的柱状体
        reward_chart = raw_env.groupby(['action', 'num']).sum()['reward'].unstack().plot.bar(title='reward')
        # 调整图例
        # reward_chart.legend(loc='right')
        reward_chart.legend(bbox_to_anchor=(1.0, 1.0))
        # 注意这里groupby(['action', 'num']) 后结果任然是有很多列，打印的时候看不出来，但操作会出错
        # 因为count()，可以随意取一列
        action_count = raw_env.groupby(['action', 'num']).count()['reward'].unstack().fillna(0)

        # action_count 进行unstack后是Dataframe，action_total则是Series
        # 所以action_total是对第二个列聚类，使得两者列能够列对齐
        action_total = raw_env.groupby(['num']).count()['action']
        freq_chart = (action_count / action_total).plot.bar(title='freq')
        freq_chart.legend(bbox_to_anchor=(1.0, 1.0))

        return raw_env

    # 测试模型数据是否匹配,只能训练之前用
    # 模型更新后就不准了
    def model_test(self, folder, round_nums):
        # 确定即时预测与回放的效果是一致的
        raw_env = self.raw_data_generate(folder, round_nums)
        train_env, train_index = self.train_env_generate(raw_env)
        train_reward, td_error, n_action = self.train_reward_generate(raw_env, train_env, train_index)

        # 检验后期生成的数据和当时采取的动作是否一样，注意比较的时候e_greedy 要设置成1,模型要没被训练过
        print(np.array(n_action[1]) == np.array(n_action[0]))
        # print(np.array(n_action[1]))

    # 输出测试，发现SeparableConv1D输出为及其稀疏,以及全连接层及其稀疏
    def output_test(self, data):
        layer_outputs = [layer.output for layer in self.predict_model.layers]
        activation_model = Model(inputs=self.predict_model.input, outputs=layer_outputs)
        layer_name = [layer.name for layer in self.predict_model.layers]
        ans = activation_model.predict(data)
        # for pair in zip(layer_name, ans):
        # print(pair)
        return {name: value for name, value in zip(layer_name, ans)}


if __name__ == '__main__':
    pass
    # 不知道为什么不再在这个文件下调用，训练的数据不对，损失极小，待查
    # model.train_folder(1, 30)
    # model.model_test()
    # data = kyo.generate_raw_train_data(3, 1)

    # print(kyo.train_model(1, 1))
    # model.train_folder(7, 1)
    # t = model.model_test(1, 1)
    # model.train_folder(10)
    # train_env_data, reward_train_data, n_action = model.generate_raw_train_data(1, 1)
    # l = model.output_test([ev[10] for ev in train_env_data])
