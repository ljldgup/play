import os
import random
from tensorflow.keras import backend as K
from kof.kof_command_mame import global_set, opposite_direction, direct_key_list, action_key_list

from tensorflow.keras import layers
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import concatenate, CuDNNLSTM, BatchNormalization
from tensorflow.python.keras.optimizers import Adam
import pandas as pd
import numpy as np

data_dir = os.getcwd()
env_colomn = ['role1_action', 'role2_action',
              'role1_energy', 'role2_energy',
              'role1_position_x', 'role1_position_y',
              'role2_position_x', 'role2_position_y', 'role1', 'guard_value',
              'role1_combo_count',
              'role1_life', 'role2_life',
              'time', 'coins', ]


# 把摇杆和按键分开
class operation_split_model():

    def __init__(self, role, model_name='operation_split_model',
                 reward_decay=0.99,
                 e_greedy=0.7,
                 # 这里把输入步数增加，并提高lua中的采样频率
                 input_steps=8,
                 reward_steps=3):
        self.action_num = len(action_key_list)
        self.direction_num = len(direct_key_list)
        self.role = role
        self.model_name = model_name
        self.reward_decay = reward_decay
        self.e_greedy = e_greedy
        # 输入步数，计算reward的步数
        self.input_steps = input_steps
        self.reward_steps = reward_steps
        global_set(role)
        self.target_model = self.build_model()
        self.predict_model = self.build_model()

        if os.path.exists('{}/model/{}_{}.index'.format(data_dir, self.role, self.model_name)):
            print('load model {}'.format(self.role))
            self.load_model()
        else:
            self.save_model()

    def build_model(self):
        role1_actions = Input(shape=(self.input_steps,), name='role1_actions')
        role2_actions = Input(shape=(self.input_steps,), name='role2_actions')
        role1_actions_embedding = layers.Embedding(512, 4, name='role1_actions_embedding')(role1_actions)
        role2_actions_embedding = layers.Embedding(512, 8, name='role2_actions_embedding')(role2_actions)

        role1_energy = Input(shape=(self.input_steps,), name='role1_energy')
        role1_energy_embedding = layers.Embedding(2, 1, name='role1_energy_embedding')(role1_energy)
        role2_energy = Input(shape=(self.input_steps,), name='role2_energy')
        role2_energy_embedding = layers.Embedding(2, 1, name='role2_energy_embedding')(role2_energy)

        role_position = Input(shape=(self.input_steps, 4), name='role_position')
        position = layers.SeparableConv1D(8, 1, padding='same', strides=1,
                                          kernel_initializer='he_uniform')(role_position)
        position = BatchNormalization()(position)
        position = layers.LeakyReLU(0.05)(position)

        direction_input = Input(shape=(self.input_steps,), name='direction_input')
        direction_input_embedding = layers.Embedding(self.action_num, 2, name='direction_input_embedding')(
            direction_input)
        action_input = Input(shape=(self.input_steps,), name='action_input')
        action_input_embedding = layers.Embedding(self.action_num, 2, name='action_input_embedding')(action_input)

        concatenated_status = concatenate(
            [role1_actions_embedding, role1_energy_embedding, role2_energy_embedding, position,
             role2_actions_embedding, direction_input_embedding, action_input_embedding
             ],
            axis=-1)
        t_status = CuDNNLSTM(512)(concatenated_status)
        t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
        t_status = layers.LeakyReLU(0.05)(t_status)

        # 线性的初始化可以用zeros
        value = layers.Dense(1, activation='linear', kernel_initializer='zeros')(t_status)
        action_value = concatenate([value] * self.action_num)
        direction_value = concatenate([value] * self.direction_num)

        action_a = layers.Dense(self.action_num, activation='linear', kernel_initializer='zeros')(t_status)
        action_mean = layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(action_a)
        action_advantage = layers.Subtract()([action_a, action_mean])
        action_q = layers.Add(name='action_q')([action_value, action_advantage])

        direction_a = layers.Dense(self.direction_num, activation='linear', kernel_initializer='zeros')(t_status)
        direction_mean = layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(direction_a)
        direction_advantage = layers.Subtract()([direction_a, direction_mean])
        direction_q = layers.Add(name='direction_q')([direction_value, direction_advantage])

        model = Model(
            [role1_actions, role2_actions, role1_energy, role2_energy,
             role_position, direction_input, action_input], [direction_q, action_q])

        model.compile(optimizer=Adam(lr=0.0001),
                      loss='mse')

        return model

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
            raw_env['direction'] = raw_actions[:, 0]
            raw_env['action'] = raw_actions[:, 1]

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
            reward = life_reward + combo_reward * 4 + 10 * energy_reward + 3 * guard_reward

            raw_env['raw_reward'] = reward
            raw_env['reward'] = reward / 80

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
                           'role2_position_x', 'role2_position_y']].loc[index - self.input_steps + 1:index]
            action = raw_env[['direction', 'action']].loc[index - self.input_steps:index - 1]

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

    def double_dqn_train_data(self, raw_env, train_env, train_index):
        reward = raw_env['reward'].reindex(train_index)
        direction = raw_env['direction'].reindex(train_index)
        direction = direction.astype('int')
        action = raw_env['action'].reindex(train_index)
        action = action.astype('int')

        target_model_prediction = self.target_model.predict(train_env)
        predict_model_prediction = self.predict_model.predict(train_env)

        pre_direction = predict_model_prediction[0].argmax(axis=1)
        pre_actions = predict_model_prediction[1].argmax(axis=1)

        # 由训练模型选动作，target模型根据动作估算q值，不关心是否最大
        # yj=Rj + γQ′(ϕ(S′j), argmaxa′Q(ϕ(S′j), a, w), w′)
        next_max_reward_direction = predict_model_prediction[0].argmax(axis=1).astype('int')
        next_max_reward_action = predict_model_prediction[1].argmax(axis=1).astype('int')

        next_direction_reward = target_model_prediction[0][range(len(train_index)), next_max_reward_direction]
        next_action_reward = target_model_prediction[1][range(len(train_index)), next_max_reward_action]

        next_direction_reward = np.roll(next_direction_reward, -1)
        next_action_reward = np.roll(next_action_reward, -1)

        time = raw_env['time'].reindex(train_index).diff(1).shift(-1).fillna(1).values
        next_direction_reward[time > 0] = 0
        next_action_reward[time > 0] = 0

        direction_reward = reward.values.copy()
        action_reward = reward.values.copy()

        direction_reward += self.reward_decay * next_direction_reward
        action_reward += self.reward_decay * next_action_reward

        predict_model_prediction[0][range(len(train_index)), direction] = reward
        predict_model_prediction[1][range(len(train_index)), action] = reward

        return [predict_model_prediction, [[pre_direction, pre_actions], [direction.values, action.values]]]

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
        self.predict_model.fit([env[random_index] for env in train_env],
                               [reward[random_index] for reward in train_reward],
                               batch_size=batch_size,
                               epochs=epochs, verbose=2)

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

    def load_model(self, name=None):
        if name:
            self.predict_model.load_weights('{}/model/{}_{}_{}'.format(data_dir, self.role, self.model_name, name))
        else:
            self.predict_model.load_weights('{}/model/{}_{}'.format(data_dir, self.role, self.model_name))
        self.weight_copy()

    def double_dqn_train_data(self, raw_env, train_env, train_index):
        reward = raw_env['reward'].reindex(train_index)
        direction = raw_env['direction'].reindex(train_index)
        direction = direction.astype('int')
        action = raw_env['action'].reindex(train_index)
        action = action.astype('int')

        target_model_prediction = self.target_model.predict(train_env)
        predict_model_prediction = self.predict_model.predict(train_env)

        pre_direction = predict_model_prediction[0].argmax(axis=1)
        pre_action = predict_model_prediction[1].argmax(axis=1)

        # 由训练模型选动作，target模型根据动作估算q值，不关心是否最大
        # yj=Rj + γQ′(ϕ(S′j), argmaxa′Q(ϕ(S′j), a, w), w′)

        next_max_reward_direction = predict_model_prediction[0].argmax(axis=1)
        next_max_reward_action = predict_model_prediction[1].argmax(axis=1)

        next_direction_reward = target_model_prediction[0][
            range(len(train_index)), next_max_reward_direction.astype('int')]
        next_action_reward = target_model_prediction[1][range(len(train_index)), next_max_reward_action.astype('int')]

        next_direction_reward = np.roll(next_direction_reward, -1)
        next_action_reward = np.roll(next_action_reward, -1)

        time = raw_env['time'].reindex(train_index).diff(1).shift(-1).fillna(1).values
        next_action_reward[time > 0] = 0
        next_direction_reward[time > 0] = 0

        direction_reward = reward.copy()
        direction_reward += self.reward_decay * next_direction_reward
        action_reward = reward.copy()
        action_reward += self.reward_decay * next_action_reward

        predict_model_prediction[0][range(len(train_index)), direction] = direction_reward
        predict_model_prediction[1][range(len(train_index)), action] = action_reward

        return [predict_model_prediction, [[pre_direction, pre_action], [direction.values, action.values]]]

    def choose_action(self, raw_data, action, random_choose=False):
        if random_choose or random.random() > self.e_greedy:
            return [random.randint(0, self.direction_num - 1), random.randint(0, self.action_num - 1)]
        else:
            # 根据dqn的理论，这里应该用训练的模型， target只在寻训练的时候使用，并且一定时间后才复制同步
            prediction = self.predict_model.predict(self.raw_env_data_to_input(raw_data, action))
            return [prediction[0].argmax(), prediction[1].argmax()]

    def raw_env_data_to_input(self, raw_data, action):
        return [raw_data[:, :, 0], raw_data[:, :, 1], raw_data[:, :, 2], raw_data[:, :, 3],
                raw_data[:, :, 4:8], action[:, :, 0], action[:, :, 1]]

    def empty_env(self):
        return [[], [], [], [], [], [], []]

    def model_test(self, folder, round_nums):
        # 确定即时预测与回放的效果是一致的
        raw_env = self.raw_data_generate(folder, round_nums)
        train_env, train_index = self.train_env_generate(raw_env)
        _, n_action = self.double_dqn_train_data(raw_env, train_env, train_index)
        print(n_action[0][0])
        print(n_action[0][0] == n_action[1][0])
        print(n_action[0][1])
        print(n_action[0][1] == n_action[1][1])
        # 重置 e_greedy
        # self.e_greedy = self.action_num / len(key_freq.keys()) * 0.8 + 0.2
        # print(t_act)
        # return [rst, t_act]


if __name__ == '__main__':
    model = operation_split_model('iori')
    # model.model_test(1, [1,2])
    # model.model_test(2, [1,2])
    #model.train_model(4, [1])
