import os
import random
import traceback
import numpy as np
from tensorflow.keras import backend as K

from kof.kof_command_mame import role_commands, global_set
from kof.kof_agent import KofAgent
from tensorflow.keras import layers
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import concatenate, BatchNormalization, CuDNNGRU, CuDNNLSTM
from tensorflow.python.keras.optimizers import Adam

'''
可调参数
输入
网络结构：距离,坐标先卷积再输入rnn效果更好
衰减因子:不能太大，或者太小，0.95左右
reward比例:不能太小，不然难以收敛
输入步数
学习率 过大的话学习效果很不好，小一点容易找到稳定的策略

改进点：
Prioritised replay
transformer
'''

data_dir = os.getcwd()


# 普通DDQN，输出分开成多个fc，再合并
# 这种方法违背网络共享信息的特点
# 位置距离卷积 + lstm + fc
class DoubleDQN(KofAgent):

    def __init__(self, role, model_name='double_dqn', reward_decay=0.98):
        super().__init__(role=role, model_name=model_name, reward_decay=reward_decay)
        self.train_reward_generate = self.double_dqn_train_data

    def build_model(self):
        role1_actions = Input(shape=(self.input_steps,), name='role1_actions')
        role2_actions = Input(shape=(self.input_steps,), name='role2_actions')
        role1_actions_embedding = layers.Embedding(512, 8, name='role1_actions_embedding')(role1_actions)
        role2_actions_embedding = layers.Embedding(512, 8, name='role2_actions_embedding')(role2_actions)

        role1_energy = Input(shape=(self.input_steps,), name='role1_energy')
        role1_energy_embedding = layers.Embedding(5, 2, name='role1_energy_embedding')(role1_energy)
        role2_energy = Input(shape=(self.input_steps,), name='role2_energy')
        role2_energy_embedding = layers.Embedding(5, 2, name='role2_energy_embedding')(role2_energy)

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
            [role1_actions_embedding, conv_position, role_distance, role1_energy_embedding,
             role2_actions_embedding, role2_energy_embedding, role2_baoqi_embedding])
        lstm_status = CuDNNLSTM(512)(concatenate_status)

        t_status = layers.Dense(256, kernel_initializer='he_uniform')(lstm_status)
        t_status = BatchNormalization()(t_status)
        t_status = layers.LeakyReLU(0.05)(t_status)
        output_layers = []
        for a in range(self.action_num):
            t_layer = layers.Dense(128, kernel_initializer='he_uniform')(t_status)
            t_layer = BatchNormalization()(t_layer)
            t_layer = layers.LeakyReLU(0.05)(t_layer)
            # 注意这里softmax 不用he_uniform初始化
            t_layer = layers.Dense(1, name='action_{}'.format(a))(t_layer)
            output_layers.append(t_layer)

        output = concatenate(output_layers)
        model = Model([role1_actions, role2_actions, role1_energy, role2_energy, role1_x_y, role2_x_y,
                       role2_baoqi], output)

        model.compile(optimizer=Adam(lr=0.00003), loss='mse')

        return model

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
        time = raw_env['time'].reindex(train_index).diff(1).shift(-1).fillna(1).values

        next_max_reward_action = predict_model_prediction.argmax(axis=1)
        next_action_reward = target_model_prediction[range(len(train_index)), next_max_reward_action.astype('int')]
        '''
        next_action_reward = np.roll(next_action_reward, -1)
        next_action_reward[time > 0] = 0
        reward += self.reward_decay * next_action_reward
        '''
        # Multi-Step Learning 一次性加上后面n步的衰减报酬
        multi_steps = 1
        for t in range(multi_steps):
            next_action_reward = np.roll(next_action_reward, -1)
            # 始终把最后一步设为0，由于移动的原因，0会被前移，所以不需要考虑之前的时间步
            next_action_reward[time > 0] = 0
            reward += self.reward_decay ** (t + 1) * next_action_reward
        # 上下限裁剪，防止过估计
        reward[reward > 1] = 1
        reward[reward < -1] = -1
        # 注意一定要取绝对值，不然很发生很严重的过估计
        td_error = abs(reward - predict_model_prediction[range(len(train_index)), action])
        td_error = td_error.values
        # 这里报action过多很可能是人物不对
        predict_model_prediction[range(len(train_index)), action] = reward

        return [predict_model_prediction, td_error, [pre_actions, action.values]]

    def raw_env_data_to_input(self, raw_data, action):
        # 这里energy改成一个只输入最后一个,这里输出的形状应该就是1，貌似在keras中也能正常运作
        return [raw_data[:, :, 0], raw_data[:, :, 1], raw_data[:, :, 2], raw_data[:, :, 3],
                raw_data[:, :, 4:6], raw_data[:, :, 6:8], raw_data[:, :, 8]]

    # 这里返回的list要和raw_env_data_to_input返回的大小一样
    def empty_env(self):
        return [[], [], [], [], [], [], []]


# 正常dueling dqn
# 将衰减降低至0.94，去掉了上次动作输入，将1p embedding带宽扩展到8，后效果比之前好了很多
# 但动作比较集中
class DuelingDQN(DoubleDQN):
    def __init__(self, role):
        super().__init__(role=role, model_name='dueling_dqn')

    def build_model(self):
        role1_actions = Input(shape=(self.input_steps,), name='role1_actions')
        role2_actions = Input(shape=(self.input_steps,), name='role2_actions')
        role1_actions_embedding = layers.Embedding(512, 8, name='role1_actions_embedding')(role1_actions)
        role2_actions_embedding = layers.Embedding(512, 8, name='role2_actions_embedding')(role2_actions)

        role1_energy = Input(shape=(self.input_steps,), name='role1_energy')
        role1_energy_embedding = layers.Embedding(5, 2, name='role1_energy_embedding')(role1_energy)
        role2_energy = Input(shape=(self.input_steps,), name='role2_energy')
        role2_energy_embedding = layers.Embedding(5, 2, name='role2_energy_embedding')(role2_energy)

        role2_baoqi = Input(shape=(self.input_steps,), name='role2_baoqi')
        role2_baoqi_embedding = layers.Embedding(2, 2, name='role2_baoqi_embedding')(role2_baoqi)

        role1_x_y = Input(shape=(self.input_steps, 2), name='role1_x_y')
        role2_x_y = Input(shape=(self.input_steps, 2), name='role2_x_y')

        role_position = concatenate([role1_x_y, role2_x_y])
        role_distance = layers.Subtract()([role1_x_y, role2_x_y])
        conv_position = layers.SeparableConv1D(8, 1, padding='same', strides=1, kernel_initializer='he_uniform')(
            role_position)
        conv_position = BatchNormalization()(conv_position)
        conv_position = layers.LeakyReLU(0.05)(conv_position)

        concatenate_status = concatenate(
            [role1_actions_embedding, role_distance, conv_position, role1_energy_embedding,
             role2_actions_embedding, role2_energy_embedding, role2_baoqi_embedding])

        lstm_status = CuDNNLSTM(512)(concatenate_status)

        t_status = layers.Dense(512, kernel_initializer='he_uniform')(lstm_status)
        t_status = layers.LeakyReLU(0.05)(t_status)
        t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
        t_status = BatchNormalization()(t_status)
        t_status = layers.LeakyReLU(0.05)(t_status)

        # 攻击动作，则采用基础标量 + 均值为0的向量策略
        value = layers.Dense(1)(t_status)
        value = concatenate([value] * self.action_num)
        a = layers.Dense(self.action_num)(t_status)
        mean = layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = layers.Subtract()([a, mean])
        q = layers.Add()([value, advantage])
        model = Model([role1_actions, role2_actions, role1_energy, role2_energy, role1_x_y, role2_x_y,
                       role2_baoqi], q)

        model.compile(optimizer=Adam(lr=0.00003), loss='mse')

        return model


# 两个角色分开做lstm
class DuelingDQN_2(DoubleDQN):

    def __init__(self, role):
        super().__init__(role=role, model_name='dueling_dqn_2')

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

        role_distance = layers.Subtract()([role1_x_y, role2_x_y])
        role_distance = BatchNormalization()(role_distance)

        lstm_role1_status = CuDNNLSTM(256)(
            concatenate([role1_actions_embedding, role_position, role_distance, role1_energy_embedding]))
        lstm_role2_status = CuDNNLSTM(256)(
            concatenate(
                [role2_actions_embedding, role_position, role_distance, role2_energy_embedding, role2_baoqi_embedding]))
        # lstm_position = CuDNNLSTM(128)(position)

        concatenated_status = concatenate(
            [lstm_role1_status, lstm_role2_status])

        t_status = layers.Dense(512, kernel_initializer='he_uniform')(concatenated_status)
        t_status = layers.LeakyReLU(0.05)(t_status)
        t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
        t_status = BatchNormalization()(t_status)
        t_status = layers.LeakyReLU(0.05)(t_status)

        value = layers.Dense(1)(t_status)
        value = concatenate([value] * self.action_num)
        a = layers.Dense(self.action_num)(t_status)
        mean = layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = layers.Subtract()([a, mean])
        q = layers.Add()([value, advantage])
        model = Model(
            [role1_actions, role2_actions, role1_energy, role2_energy, role1_x_y, role2_x_y,
             role2_baoqi],
            q)

        model.compile(optimizer=Adam(lr=0.00003), loss='mse')

        return model


if __name__ == '__main__':
    models = [DuelingDQN('iori'), DoubleDQN('iori'), DuelingDQN_2('iori')]
    # model = DuelingDQN('iori')

    # model.model_test(1, [1,2])
    # model.model_test(2, [1,2])
    # model.predict_model.summary()
    # t = model.operation_analysis(5)

    for model in models:
        print('-----------------------------')
        print('train ', model.model_name)
        for i in range(1, 14):
            try:
                print('train ', i)
                for r in range(1, 4):
                    # model.train_model(i, [r])
                    model.train_model_with_sum_tree(i, [r])
                    model.weight_copy()
            except:
                traceback.print_exc()
        model.save_model()

    for model in models:
        print('-----------------------------')
        print('test ', model.model_name)
        model.model_test(12, [1])
    '''
    '''
    '''
    raw_env = model.raw_data_generate(2, [1])
    train_env, train_index = model.train_env_generate(raw_env)
    train_reward, n_action = model.double_dqn_train_data(raw_env, train_env, train_index)
    t = model.predict_model.predict([np.expand_dims(env[100], 0) for env in train_env])
    # output = model.output_test([ev[50].reshape(1, *ev[50].shape) for ev in train_env])

    model.model_test(12, [1])
    '''
