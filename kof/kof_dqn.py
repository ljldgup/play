import math
import os
import random
import time

import pandas as pd
import numpy as np
from tensorflow.python.keras import Model

from kof.kof_command_mame import global_set, role_commands

'''
# 把图像拷贝后重新训练一下
# blood_clf = if_blood_train()
# blood_clf = read_simple_model('svm_blood_judge')
'''
data_dir = os.getcwd()
env_colomn = ['role1_action', 'role2_action',
              'role1_energy', 'role2_energy',
              'role1_position_x', 'role1_position_y',
              'role2_position_x', 'role2_position_y', 'baoqi', 'role1', 'role2', 'guard_value',
              'role1_combo_count',
              'role1_life', 'role2_life',
              'time', 'coins', ]


class kof_dqn():
    def __init__(self, role, model_name,
                 reward_decay=0.98,
                 e_greedy=0.7,
                 input_steps=3,
                 reward_steps=3):
        self.role = role
        self.model_name = model_name
        self.reward_decay = reward_decay
        self.e_greedy = e_greedy
        # 输入步数，计算reward的步数
        self.input_steps = input_steps
        self.reward_steps = reward_steps
        global_set(role)
        self.action_num = len(role_commands[role])
        self.target_model = self.build_model()
        self.predict_model = self.build_model()
        self.record = self.get_record()
        if os.path.exists('{}/model/{}_{}.index'.format(data_dir, self.role, self.model_name)):
            print('load model {}'.format(self.role))
            self.load_model()
        else:
            self.save_model()

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

            # 避免浪费大招
            energy_reward = raw_env['role1_energy'].diff(1).fillna(0)
            energy_reward = energy_reward.map(lambda x: 0 if x > 0 else x)

            # 防守的收益
            guard_reward = -raw_env['guard_value'].diff(1).fillna(0)
            guard_reward = guard_reward.map(lambda x: x if x > 0 else 0)

            # 连招收益
            combo_reward = raw_env['role1_combo_count'].diff(1).fillna(0)
            # 由guard，energy_reward，另外几个基本不会并存
            reward = life_reward + 5 * energy_reward + guard_reward

            # 生成time_steps时间步内的reward
            # 改成dqn 因为自动加后面一次报酬，后应该不需要rolling
            # reward_sum = reward.rolling(self.reward_steps, min_periods=1).sum().shift(-self.reward_steps).fillna(0)

            # reward_sum太小无法收敛，太大则整体不稳定
            raw_env['raw_reward'] = reward
            raw_env['reward'] = reward / 40

            # 使用log(n+x)-log(n)缩放reward，防止少量特别大的动作影响收敛，目前来看缩放的越小，收敛效果越好。
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
                           'role2_position_x', 'role2_position_y', 'baoqi']].loc[index - self.input_steps + 1:index]
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

    # 最开始用的不考虑长期收益的模型，在pandas钟使用rolling来使其拥有少量的长期收益
    # reward中无关动作随着模型改变而改变
    # 为了便于重新生成，将其分离
    def train_reward_generate(self, raw_env, train_env, train_index):

        # 筛选并调整reward顺序
        reward = raw_env['reward'].reindex(train_index)
        action = raw_env['action'].reindex(train_index)
        action = action.astype('int')
        # 注意这里使用预测模型输出作为y，否则会导致模型不稳定
        prediction = self.predict_model.predict(train_env)
        # 用来确认生成对象是否正确
        pre_actions = prediction.argmax(axis=1)
        # pre_prediction = prediction.copy().

        # 注意这里通过[[...],[...]]直接选择里局矩阵里的多个位置，并进行赋值
        # 这里直接给pandas series的格式也可以，可迭代貌似就行
        prediction[range(len(train_index)), action] = reward

        return [prediction, [pre_actions, action.values]]

    def nature_dqn_reward_generate(self, raw_env, train_env, train_index):
        reward = raw_env['reward'].reindex(train_index)
        action = raw_env['action'].reindex(train_index)
        action = action.astype('int')

        target_model_prediction = self.target_model.predict(train_env)
        predict_model_prediction = self.predict_model.predict(train_env)
        # pre_prediction = predict_model_prediction.copy()
        pre_actions = predict_model_prediction.argmax(axis=1)

        # yj=Rj+γmaxa′Q′(ϕ(S′j),A′j,w′)
        # 下一步的最大报酬
        next_action_reward = target_model_prediction.max(axis=1)
        next_action_reward = np.roll(next_action_reward, -1)

        # 比上次时间大，说明重来了，上次就是end，fillna用于填充结尾，结尾也是end
        time = raw_env['time'].reindex(train_index).diff(1).shift(-1).fillna(1).values
        # 最后一此操作，下一次的报酬为0
        next_action_reward[time > 0] = 0
        reward += next_action_reward * self.reward_decay
        predict_model_prediction[range(len(train_index)), action.values] = reward

        return [predict_model_prediction, [pre_actions, action]]

    def double_dqn_train_data(self, raw_env, train_env, train_index):
        reward = raw_env['reward'].reindex(train_index)
        action = raw_env['action'].reindex(train_index)
        action = action.astype('int')

        target_model_prediction = self.target_model.predict(train_env)
        predict_model_prediction = self.predict_model.predict(train_env)
        # pre_prediction = predict_model_prediction.copy()
        pre_actions = predict_model_prediction.argmax(axis=1)

        # 由训练模型选动作，target模型根据动作估算q值，不关心是否最大
        # yj=Rj + γQ′(ϕ(S′j), argmaxa′Q(ϕ(S′j), a, w), w′)
        next_max_reward_action = predict_model_prediction.argmax(axis=1)
        next_action_reward = target_model_prediction[range(len(train_index)), next_max_reward_action.astype('int')]

        next_action_reward = np.roll(next_action_reward, -1)
        time = raw_env['time'].reindex(train_index).diff(1).shift(-1).fillna(1).values
        next_action_reward[time > 0] = 0
        reward += self.reward_decay * next_action_reward

        # 这里报action过多很可能是人物不对
        predict_model_prediction[range(len(train_index)), action] = reward

        return [predict_model_prediction, [pre_actions, action.values]]

    # 在线学习可以batch_size设置成1
    # 另外随机打乱了以后训练几乎无法收敛，即使是batch_size==1的情况
    # 改成所有数据何在一起，打乱顺序，使用batch 训练，速度快了很多
    # batch_size的选取不同，损失表现完全不一样
    def train_model(self, folder, round_nums=[], batch_size=32, epochs=30):
        if not round_nums:
            round_nums = list(set(
                [file.split('.')[0] for file in os.listdir('{}/{}'.format(data_dir, folder))]
            ))

        raw_env = self.raw_data_generate(folder, round_nums)
        train_env, train_index = self.train_env_generate(raw_env)
        train_reward, action = self.double_dqn_train_data(raw_env, train_env, train_index)

        random_index = np.random.permutation(len(train_index))
        # verbose参数控制输出，这里每个epochs输出一次
        self.predict_model.fit([env[random_index] for env in train_env], train_reward[random_index],
                               batch_size=batch_size,
                               epochs=epochs, verbose=2)

        self.record['total_epochs'] += epochs

    def weight_copy(self):
        self.target_model.set_weights(self.predict_model.get_weights())

    def save_model(self, name=None):
        if not os.path.exists('{}/model'.format(data_dir)):
            os.mkdir('{}/model'.format(data_dir))
        self.weight_copy()
        if name:
            # 用于临时保存表现最好的模型
            self.target_model.save_weights('{}/model/{}_{}_{}'.format(data_dir, self.role, self.model_name, name))
        else:
            self.target_model.save_weights('{}/model/{}_{}'.format(data_dir, self.role, self.model_name))

        with open('{}/model/{}_{}_record'.format(data_dir, self.role, self.model_name), 'w') as r:
            r.write(str(self.record))

    def load_model(self, name=None):
        if name:
            self.predict_model.load_weights('{}/model/{}_{}_{}'.format(data_dir, self.role, self.model_name, name))
        else:
            self.predict_model.load_weights('{}/model/{}_{}'.format(data_dir, self.role, self.model_name))
        self.weight_copy()

    def build_model(self):
        pass

    # 把原始环境输入，分割成模型能接受的输入，在具体的模型中实现
    def raw_env_data_to_input(self, raw_data, action):
        pass

    def empty_env(self):
        pass

    def get_record(self):
        if os.path.exists('{}/model/{}_{}_record'.format(data_dir, self.role, self.model_name)):
            with open('{}/model/{}_{}_record'.format(data_dir, self.role, self.model_name), 'r') as r:
                record = eval(r.read())
        else:
            record = {}
            record['total_epochs'] = 0
            record['begin_time'] = time.asctime(time.localtime(time.time()))
        return record

    # 分析动作的出现频率和回报率
    def operation_analysis(self, folder):
        round_nums = list(set(
            [file.split('.')[0] for file in os.listdir('{}/{}'.format(data_dir, folder))]
        ))

        raw_env = self.raw_data_generate(folder, [round_nums[0]])
        raw_env['num'] = round_nums[0]
        for i in range(1, len(round_nums)):
            t = self.raw_data_generate(folder, [round_nums[i]])
            t['num'] = round_nums[i]
            raw_env = pd.concat([raw_env, t])

        # 注意这里对num，action进行聚类，之后他们都在层次化索引上
        # 所以需要unstack，将action移到列名,才能绘制出按文件名分开的柱状体
        reward_chart = raw_env.groupby(['num', 'action']).sum()['reward'].unstack().plot.bar(title='reward')
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

        for action in action_count:
            action_count[action]

        return raw_env

    # 测试模型数据是否匹配，
    # 测试模型输出是否过于单一
    def model_test(self, folder, round_nums):
        # 确定即时预测与回放的效果是一致的
        raw_env = self.raw_data_generate(folder, round_nums)
        train_env, train_index = self.train_env_generate(raw_env)
        _, n_action = self.train_reward_generate(raw_env, train_env, train_index)
        # 检验后期生成的数据和当时采取的动作是否一样，注意比较的时候e_greedy 要设置成1
        print(np.array(n_action[1]) == np.array(n_action[0]))
        print(np.array(n_action[1]))

        t_env = raw_env.values
        t_act = raw_env['action'].values
        t_env = t_env.reshape(1, *t_env.shape)
        t_act = t_act.reshape(1, *t_act.shape)
        rst = []
        for i in range(1, len(t_env[0]) - self.input_steps - 1):
            t1 = t_env[:, i:i + self.input_steps, :]
            t2 = t_act[:, i - 1:i + self.input_steps - 1]
            # print(self.raw_env_data_to_input(t1, t2))
            ans = self.predict_model.predict(
                self.raw_env_data_to_input(t1, t2))
            # print(ans)
            rst.append(ans.argmax())
        # print('------')
        # print(ans[:, ans.argmax()])
        # print(rst)

        # 统计频率

        key_freq = {key: 0 for key in set(rst)}
        print(key_freq.keys())
        for key in rst:
            key_freq[key] += 1
        # 各个命令的百分比
        for key in key_freq.keys():
            print('{} {:.2f}%'.format(key, key_freq[key] / len(rst) * 100))
        # 重置 e_greedy
        # self.e_greedy = self.action_num / len(key_freq.keys()) * 0.8 + 0.2
        # print(t_act)
        # return [rst, t_act]

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
