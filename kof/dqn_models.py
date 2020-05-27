import os
import random
import traceback

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
        self.model_name = 'random'
        self.role = role
        self.input_steps = 3
        self.e_greedy = 0
        self.action_num = len(role_commands[role])

    def choose_action(self, raw_data, action, random_choose=False):
        return random.randint(0, self.action_num - 1)

    def save_model(self):
        pass


# 修改dueling dqn,是的value分成防守，闪避，攻击，并在攻击value上加攻击动作advantage
class model_1(kof_dqn):
    def __init__(self, role, ):
        super().__init__(role=role, model_name='m1', input_steps=6, reward_steps=4, e_greedy=0.7, reward_decay=0.98)
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
        conv_distance = layers.SeparableConv1D(4, 1, padding='same', strides=1, kernel_initializer='he_uniform')(
            role_distance)
        conv_distance = BatchNormalization()(conv_distance)
        conv_distance = layers.LeakyReLU(0.05)(conv_distance)

        action_input = Input(shape=(self.input_steps,), name='action_input')
        action_input_embedding = layers.Embedding(self.action_num, 4, name='action_input_embedding')(action_input)

        concatenate_status = concatenate(
            [role1_actions_embedding, conv_position, conv_distance, action_input_embedding, role1_energy_embedding,
             role2_actions_embedding, role2_energy_embedding, role2_baoqi_embedding])
        lstm_status = CuDNNLSTM(512)(concatenate_status)

        t_status = layers.Dense(512, kernel_initializer='he_uniform')(lstm_status)
        t_status = layers.LeakyReLU(0.05)(t_status)
        t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
        t_status = BatchNormalization()(t_status)
        t_status = layers.LeakyReLU(0.05)(t_status)

        base_action_type = 4
        guard = layers.Dense(base_action_type - 1)(t_status)

        value = layers.Dense(1)(t_status)
        value = concatenate([value] * (self.action_num - base_action_type + 1))
        a = layers.Dense(self.action_num - base_action_type + 1)(t_status)
        mean = layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = layers.Subtract()([a, mean])
        attack = layers.Add()([value, advantage])
        output = concatenate([guard, attack])
        model = Model(
            [role1_actions, role2_actions, role1_energy, role2_energy, role1_x_y, role2_x_y,
             role2_baoqi, action_input], output)

        model.compile(optimizer=Adam(lr=0.00003), loss='mse')

        return model

    def raw_env_data_to_input(self, raw_data, action):
        return [raw_data[:, :, 0], raw_data[:, :, 1], raw_data[:, :, 2], raw_data[:, :, 3],
                raw_data[:, :, 4:6], raw_data[:, :, 6:8], raw_data[:, :, 8], action]

    # 这里返回的list要和raw_env_data_to_input返回的大小一样
    def empty_env(self):
        return [[], [], [], [], [], [], [], []]


# 两个角色分开做lstm
class model_2(kof_dqn):

    def __init__(self, role, ):
        super().__init__(role=role, model_name='m2', input_steps=6, reward_steps=4, e_greedy=0.7,
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
        conv_distance = layers.SeparableConv1D(4, 1, padding='same', strides=1, kernel_initializer='he_uniform')(
            role_distance)
        conv_distance = BatchNormalization()(conv_distance)
        conv_distance = layers.LeakyReLU(0.05)(conv_distance)

        action_input = Input(shape=(self.input_steps,), name='action_input')
        action_input_embedding = layers.Embedding(self.action_num, 4, name='action_input_embedding')(action_input)

        lstm_role1_status = CuDNNLSTM(256)(
            concatenate([role1_actions_embedding, conv_position, conv_distance, action_input_embedding,
                         role1_energy_embedding]))
        lstm_role2_status = CuDNNLSTM(256)(
            concatenate(
                [role2_actions_embedding, conv_position, conv_distance, role2_energy_embedding, role2_baoqi_embedding]))
        # lstm_position = CuDNNLSTM(128)(position)

        concatenated_status = concatenate(
            [lstm_role1_status, lstm_role2_status])

        t_status = layers.Dense(512, kernel_initializer='he_uniform')(concatenated_status)
        t_status = layers.LeakyReLU(0.05)(t_status)
        t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
        t_status = BatchNormalization()(t_status)
        t_status = layers.LeakyReLU(0.05)(t_status)

        # 这里对dueling_dqn_model做一些修改，防守进攻作为基础value，进攻优势估计加在进攻value之上
        # 攻击动作，则采用基础标量 + 均值为0的向量策略
        # 基本动作类型，前几种是防御或者闪避，最后是攻击
        base_action_type = 4
        guard = layers.Dense(base_action_type - 1)(t_status)
        # 进攻
        value = layers.Dense(1)(t_status)
        value = concatenate([value] * (self.action_num - base_action_type + 1))
        a = layers.Dense(self.action_num - base_action_type + 1)(t_status)
        mean = layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = layers.Subtract()([a, mean])
        attack = layers.Add()([value, advantage])
        output = concatenate([guard, attack])
        model = Model(
            [role1_actions, role2_actions, role1_energy, role2_energy, role1_x_y, role2_x_y,
             role2_baoqi, action_input],
            output)

        model.compile(optimizer=Adam(lr=0.00003), loss='mse')

        return model

    def raw_env_data_to_input(self, raw_data, action):
        return [raw_data[:, :, 0], raw_data[:, :, 1], raw_data[:, :, 2], raw_data[:, :, 3],
                raw_data[:, :, 4:6], raw_data[:, :, 6:8], raw_data[:, :, 8], action]

    # 这里返回的list要和raw_env_data_to_input返回的大小一样
    def empty_env(self):
        return [[], [], [], [], [], [], [], []]


# model_2基础上增加一个 lstm
class model_3(kof_dqn):

    def __init__(self, role, ):
        super().__init__(role=role, model_name='m3', input_steps=6, reward_steps=4, e_greedy=0.7,
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
        conv_distance = layers.SeparableConv1D(4, 1, padding='same', strides=1, kernel_initializer='he_uniform')(
            role_distance)
        conv_distance = BatchNormalization()(conv_distance)
        conv_distance = layers.LeakyReLU(0.05)(conv_distance)

        action_input = Input(shape=(self.input_steps,), name='action_input')
        action_input_embedding = layers.Embedding(self.action_num, 4, name='action_input_embedding')(action_input)

        lstm_role1_status = CuDNNLSTM(256, return_sequences=True)(
            concatenate([role1_actions_embedding, conv_position, conv_distance, action_input_embedding,
                         role1_energy_embedding]))
        lstm_role2_status = CuDNNLSTM(256, return_sequences=True)(
            concatenate(
                [role2_actions_embedding, conv_position, conv_distance, role2_energy_embedding, role2_baoqi_embedding]))
        # lstm_position = CuDNNLSTM(128)(position)

        lstm_status = CuDNNLSTM(512)(concatenate([lstm_role1_status, lstm_role2_status]))
        t_status = layers.Dense(512, kernel_initializer='he_uniform')(lstm_status)
        t_status = layers.LeakyReLU(0.05)(t_status)
        t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
        t_status = BatchNormalization()(t_status)
        t_status = layers.LeakyReLU(0.05)(t_status)

        base_action_type = 4
        guard = layers.Dense(base_action_type - 1)(t_status)
        # 进攻
        value = layers.Dense(1)(t_status)
        value = concatenate([value] * (self.action_num - base_action_type + 1))
        a = layers.Dense(self.action_num - base_action_type + 1)(t_status)
        mean = layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = layers.Subtract()([a, mean])
        attack = layers.Add()([value, advantage])
        output = concatenate([guard, attack])
        model = Model(
            [role1_actions, role2_actions, role1_energy, role2_energy, role1_x_y, role2_x_y,
             role2_baoqi, action_input],
            output)

        model.compile(optimizer=Adam(lr=0.00005), loss='mse')

        return model

    def raw_env_data_to_input(self, raw_data, action):
        return [raw_data[:, :, 0], raw_data[:, :, 1], raw_data[:, :, 2], raw_data[:, :, 3],
                raw_data[:, :, 4:6], raw_data[:, :, 6:8], raw_data[:, :, 8], action]

    def empty_env(self):
        return [[], [], [], [], [], [], [], []]



# 位置距离卷积 + lstm + 正常dueling dqn
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

        action_input = Input(shape=(self.input_steps,), name='action_input')
        action_input_embedding = layers.Embedding(self.action_num, 4, name='action_input_embedding')(action_input)
        concatenate_status = concatenate(
            [role1_actions_embedding, role_distance, conv_position, action_input_embedding, role1_energy_embedding,
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
        model = Model(
            [role1_actions, role2_actions, role1_energy, role2_energy, role1_x_y, role2_x_y,
             role2_baoqi, action_input],
            q)

        model.compile(optimizer=Adam(lr=0.00003), loss='mse')

        return model

    def raw_env_data_to_input(self, raw_data, action):
        # 这里energy改成一个只输入最后一个,这里输出的形状应该就是1，貌似在keras中也能正常运作
        return [raw_data[:, :, 0], raw_data[:, :, 1], raw_data[:, :, 2], raw_data[:, :, 3],
                raw_data[:, :, 4:6], raw_data[:, :, 6:8], raw_data[:, :, 8], action]

    # 这里返回的list要和raw_env_data_to_input返回的大小一样
    def empty_env(self):
        return [[], [], [], [], [], [], [], []]


if __name__ == '__main__':
    model = model_3('iori')
    # model = model_3('iori')
    # model.model_test(1, [1,2])
    # model.model_test(2, [1,2])

    # t = model.operation_analysis(5)

    for i in range(3):
        try:
            for j in range(2):
                print('train ', i)
                model.train_model(i, epochs=40)
                model.weight_copy()
        except:
            print('no data in ', i)
            # traceback.print_exc()
    '''
    raw_env = model.raw_data_generate(1, [1])
    train_env, train_index = model.train_env_generate(raw_env)
    train_reward, n_action = model.double_dqn_train_data(raw_env, train_env, train_index)
    # output = model.output_test([ev[50].reshape(1, *ev[50].shape) for ev in train_env])
    '''
