import os
import random
from tensorflow.keras import backend as K

from kof.kof_command_mame import role_commands
from kof.kof_dqn import kof_dqn
from tensorflow.keras import layers
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import concatenate, BatchNormalization, CuDNNGRU, CuDNNLSTM
from tensorflow.python.keras.optimizers import Adam

data_dir = os.getcwd()


class random_model():
    def __init__(self, role):
        self.role = role
        self.input_steps = 3
        self.e_greedy = 0
        self.action_num = len(role_commands[role])

    def choose_action(self, raw_data, action, random_choose=False):
        return random.randint(0, self.action_num - 1)

    def save_model(self):
        pass


# 这里改成 1d 卷积 + lstm + dueling dqn
class model_1(kof_dqn):

    def __init__(self, role, ):
        super().__init__(role=role, model_name='m1', input_steps=6, reward_steps=4, e_greedy=0.7)
        if os.path.exists('{}/{}_{}.index'.format(data_dir, self.role, self.model_name)):
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
        role1_energy_embedding = layers.Embedding(5, 2, name='role1_energy_embedding')(role1_energy)
        role2_energy = Input(shape=(self.input_steps,), name='role2_energy')
        role2_energy_embedding = layers.Embedding(5, 2, name='role2_energy_embedding')(role2_energy)

        role_position = Input(shape=(self.input_steps, 4), name='role_position')

        action_input = Input(shape=(self.input_steps,), name='action_input')
        action_input_embedding = layers.Embedding(self.action_num, 4, name='action_input_embedding')(action_input)
        concatenate_status = concatenate(
            [role1_actions_embedding, role_position, action_input_embedding, role1_energy_embedding,
             role2_actions_embedding, role2_energy_embedding])

        conv_status = layers.SeparableConv1D(32, 1, padding='same', strides=1, kernel_initializer='he_uniform')(
            concatenate_status)
        conv_status = BatchNormalization()(conv_status)
        conv_status = layers.LeakyReLU(0.05)(conv_status)
        conv_status = layers.SeparableConv1D(64, 2, padding='same', strides=1, kernel_initializer='he_uniform')(
            conv_status)
        conv_status = BatchNormalization()(conv_status)
        conv_status = layers.LeakyReLU(0.05)(conv_status)
        conv_status = layers.MaxPool1D(2, 1, padding='same')(conv_status)
        lstm_status = CuDNNLSTM(256)(conv_status)
        # lstm_position = CuDNNLSTM(128)(position)

        t_status = layers.Dense(256, kernel_initializer='he_uniform')(lstm_status)
        t_status = layers.LeakyReLU(0.05)(t_status)

        # 这里对dueling_dqn_model做一些修改，防守单独输出
        # 攻击动作，则采用基础标量 + 均值为0的向量策略
        guard = layers.Dense(2)(t_status)
        # 线性的初始化可以用zeros
        value = layers.Dense(1)(t_status)
        value = concatenate([value] * (self.action_num - 2))
        a = layers.Dense(self.action_num - 2)(t_status)
        mean = layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = layers.Subtract()([a, mean])
        attack = layers.Add()([value, advantage])
        output = concatenate([guard, attack])
        model = Model(
            [role1_actions, role2_actions, role1_energy, role2_energy,
             role_position, action_input], output)

        model.compile(optimizer=Adam(lr=0.0001), loss='mse')

        return model

    def raw_env_data_to_input(self, raw_data, action):
        # 这里energy改成一个只输入最后一个,这里输出的形状应该就是1，貌似在keras中也能正常运作
        return [raw_data[:, :, 0], raw_data[:, :, 1], raw_data[:, :, 2], raw_data[:, :, 3],
                raw_data[:, :, 4:8], action]

    def empty_env(self):
        return [[], [], [], [], [], []]


# 对相近的特征先做卷积，再合并处理
# 增加bn层加快收敛
class model_2(kof_dqn):
    def __init__(self, role):
        super().__init__(role=role, model_name='m2')
        if os.path.exists('{}/{}_{}.index'.format(data_dir, self.role, self.model_name)):
            print('load model {}'.format(self.role))
            self.load_model()
        else:
            self.save_model()

    def build_model(self):
        # 角色，动作，能量同样类型共用一个embedding层
        role_embedding = layers.Embedding(40, 4)
        role1 = Input(shape=(self.input_steps,), name='role1')
        role2 = Input(shape=(self.input_steps,), name='role2')
        role1_embedding = role_embedding(role1)
        role2_embedding = role_embedding(role2)

        # 由于要保留时间不，Embedding只能输入2维度，所以分开输入
        actions_embedding = layers.Embedding(512, 8, name='role_actions_embedding')
        role1_actions = Input(shape=(self.input_steps,), name='role1_actions')
        role2_actions = Input(shape=(self.input_steps,), name='role2_actions')
        role1_actions_embedding = actions_embedding(role1_actions)
        role2_actions_embedding = actions_embedding(role2_actions)

        energy_embedding = layers.Embedding(6, 2, name='role_energy')
        role1_energy = Input(shape=(self.input_steps,), name='role1_energy')
        role2_energy = Input(shape=(self.input_steps,), name='role2_energy')
        role1_energy_embedding = energy_embedding(role1_energy)
        role2_energy_embedding = energy_embedding(role2_energy)

        action_input = Input(shape=(self.input_steps,), name='action_input')
        action_input_embedding = layers.Embedding(25, 3, name='action_input_embedding')(action_input)

        role_position = Input(shape=(self.input_steps, 4), name='role_position')

        # 按照时间轴进行拼接
        # 这里尝试多加几次role_embedding，增加选角对行动的影响
        concatenated_role_actions = concatenate([role1_embedding, role2_embedding,
                                                 role1_actions_embedding, role2_actions_embedding],
                                                axis=-1)

        concatenated_role_position = concatenate([role1_embedding, role2_embedding,
                                                  role_position],
                                                 axis=-1)

        concatenated_others = concatenate([role1_embedding, role2_embedding,
                                           role1_energy_embedding, role2_energy_embedding,
                                           action_input_embedding],
                                          axis=-1)

        t_actions = layers.Conv1D(32, 1, padding='same', strides=1)(concatenated_role_actions)
        t_actions = layers.ReLU(t_actions)
        t_actions = layers.Conv1D(64, 2, padding='same', strides=1)(t_actions)
        t_actions = layers.ReLU(t_actions)
        t_actions = layers.Conv1D(128, 2, padding='same', strides=1)(t_actions)
        t_actions = BatchNormalization()(t_actions)
        t_actions = layers.ReLU(t_actions)

        t_position = layers.Conv1D(16, 1, padding='same', strides=1)(concatenated_role_position)
        t_position = layers.ReLU(t_position)
        t_position = layers.Conv1D(32, 2, padding='same', strides=1)(t_position)
        t_position = layers.ReLU(t_position)
        t_position = layers.Conv1D(64, 2, padding='same', strides=1)(t_position)
        t_position = BatchNormalization()(t_position)
        t_position = layers.ReLU(t_position)

        t_others = layers.Conv1D(16, 1, padding='same', strides=1)(concatenated_others)
        t_others = layers.ReLU(t_others)
        t_others = layers.Conv1D(32, 2, padding='same', strides=1)(t_others)
        t_others = layers.ReLU(t_others)
        t_others = layers.Conv1D(64, 2, padding='same', strides=1)(t_others)
        t_others = BatchNormalization()(t_others)
        t_others = layers.ReLU(t_others)

        t_status = concatenate([t_actions, t_position, t_others], axis=-1)
        t_status = layers.Conv1D(320, 2, padding='same', strides=1)(t_status)
        t_status = BatchNormalization()(t_status)
        t_status = layers.ReLU(t_status)

        t_status = CuDNNGRU(256)(t_status)
        t_status = layers.Dropout(0.4)(t_status)
        t_status = layers.Dense(256, kernel_initializer='he_uniform')(t_status)
        t_status = BatchNormalization()(t_status)
        t_status = layers.ReLU(t_status)
        t_status = layers.Dense(196, kernel_initializer='he_uniform')(t_status)
        t_status = layers.ReLU(t_status)

        output_operation = layers.Dense(self.action_num)(t_status)
        model = Model([role1, role2, role1_actions, role2_actions, role1_energy, role2_energy,
                       role_position, action_input], output_operation)

        # 学习率需要根据reward的尺度反复调，reward 变大，尺度也需要变大，可以只用train_folder函数进行测试
        model.compile(optimizer=Adam(lr=0.001),
                      loss='mse',
                      metrics=['acc'])

        return model

    def raw_env_data_to_input(self, raw_data, action):
        return [raw_data[:, -1, 0], raw_data[:, -1, 1],
                raw_data[:, -1, 2], raw_data[:, -1, 3],
                raw_data[:, -1, 4], raw_data[:, -1, 5],
                raw_data[:, -1, 6:10], action]

    def empty_env(self):
        return [[], [], [], [], [], [], [], []]


class dueling_dqn_model(kof_dqn):

    def __init__(self, role, ):
        super().__init__(role=role, model_name='dueling_dqn', input_steps=6, reward_steps=4, e_greedy=0.7,
                         reward_decay=0.98)
        if os.path.exists('{}/{}_{}.index'.format(data_dir, self.role, self.model_name)):
            print('load model {}'.format(self.role))
            self.load_model()
        else:
            self.save_model()

    def build_model(self):
        role1_actions = Input(shape=(self.input_steps,), name='role1_actions')
        role2_actions = Input(shape=(self.input_steps,), name='role2_actions')
        role1_actions_embedding = layers.Embedding(512, 4, name='role1_actions_embedding')(role1_actions)
        role2_actions_embedding = layers.Embedding(512, 8, name='role2_actions_embedding')(role2_actions)

        role1_energy = Input(shape=(1,), name='role1_energy')
        role1_energy_embedding = layers.Embedding(5, 2, name='role1_energy_embedding')(role1_energy)
        role2_energy = Input(shape=(1,), name='role2_energy')
        role2_energy_embedding = layers.Embedding(5, 2, name='role2_energy_embedding')(role2_energy)

        role_position = Input(shape=(self.input_steps, 4), name='role_position')

        action_input = Input(shape=(self.input_steps,), name='action_input')
        action_input_embedding = layers.Embedding(self.action_num, 4, name='action_input_embedding')(action_input)

        lstm_role1_action = CuDNNLSTM(64)(
            concatenate([role1_actions_embedding, role_position, action_input_embedding]))
        lstm_role2_action = CuDNNLSTM(64)(concatenate([role2_actions_embedding, role_position]))
        # lstm_position = CuDNNLSTM(128)(position)

        concatenated_status = concatenate(
            [lstm_role1_action, lstm_role2_action,
             layers.Flatten()(role1_energy_embedding), layers.Flatten()(role2_energy_embedding)])
        t_status = layers.Dense(256, kernel_initializer='he_uniform')(concatenated_status)
        t_status = layers.LeakyReLU(0.05)(t_status)
        t_status = layers.Dense(256, kernel_initializer='he_uniform')(t_status)
        t_status = BatchNormalization()(t_status)
        t_status = layers.LeakyReLU(0.05)(t_status)

        # 这里对dueling_dqn_model做一些修改，防守进攻作为基础value，进攻优势估计加在进攻value之上
        # 攻击动作，则采用基础标量 + 均值为0的向量策略
        guard = layers.Dense(2)(t_status)
        # 进攻
        value = layers.Dense(1)(t_status)
        value = concatenate([value] * (self.action_num - 2))
        a = layers.Dense(self.action_num - 2)(t_status)
        mean = layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = layers.Subtract()([a, mean])
        attack = layers.Add()([value, advantage])
        output = concatenate([guard, attack])
        model = Model(
            [role1_actions, role2_actions, role1_energy, role2_energy, role_position, action_input], output)

        model.compile(optimizer=Adam(lr=0.0001), loss='mse')

        return model

    def raw_env_data_to_input(self, raw_data, action):
        # 这里energy改成一个只输入最后一个,这里输出的形状应该就是1，貌似在keras中也能正常运作
        return [raw_data[:, :, 0], raw_data[:, :, 1], raw_data[:, -1, 2], raw_data[:, -1, 3],
                raw_data[:, :, 4:8], action]

    def empty_env(self):
        return [[], [], [], [], [], []]


if __name__ == '__main__':
    model = dueling_dqn_model('kyo')
    # model.model_test(1, [1,2])
    # model.model_test(2, [1,2])

    for i in range(1):
        model.train_model(2, epochs=50)
        model.weight_copy()
    '''
    raw_env = model.raw_data_generate(2, [2])
    train_env, train_index = model.train_env_generate(raw_env)
    train_reward, n_action = model.train_reward_generate(raw_env, train_env, train_index)
    # output = model.output_test([ev[50].reshape(1, *ev[50].shape) for ev in train_env])
   '''
