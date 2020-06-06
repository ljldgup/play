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
from tensorflow.keras import losses

from kof.value_based_models import DuelingDQN

data_dir = os.getcwd()


class ActorCritic(KofAgent):

    def __init__(self, role, model_name='ActorCritic'):
        super().__init__(role=role, model_name=model_name)
        self.critic = DuelingDQN(role, model_name="critic")
        self.actor = self.predict_model
        self.train_reward_generate = self.actor_tarin_data

    def choose_action(self, raw_data, action, random_choose=False):
        if random_choose or random.random() > self.e_greedy:
            return random.randint(0, self.action_num - 1)
        else:
            return self.actor.predict(self.raw_env_data_to_input(raw_data, action)).argmax()

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

        t_status = layers.Dense(512, kernel_initializer='he_uniform')(lstm_status)
        t_status = BatchNormalization()(t_status)
        t_status = layers.LeakyReLU(0.05)(t_status)
        output = layers.Dense(self.action_num, activation='softmax')(t_status)

        model = Model([role1_actions, role2_actions, role1_energy, role2_energy, role1_x_y, role2_x_y,
                       role2_baoqi], output)

        model.compile(optimizer=Adam(lr=0.00001), loss=losses.categorical_crossentropy)

        return model

    def actor_tarin_data(self, raw_env, train_env, train_index):
        _, td_error, action = self.critic.train_reward_generate(raw_env, train_env, train_index)

        # 构造onehot，将1改为td_error,
        # 使用mean(td_error * (log(action_prob)))，用于训练
        action_onehot = np.zeros(shape=(len(action[1]), self.action_num))
        action_onehot[range(len(action[1])), action[1]] = td_error
        return action_onehot, td_error, action

    def train_model_with_sum_tree(self, folder, round_nums=[], batch_size=64, epochs=30):
        # 先使用critic的td_error交叉熵训练actor，再训练critic
        KofAgent.train_model_with_sum_tree(self, folder, round_nums=[], batch_size=batch_size, epochs=epochs)
        self.critic.train_model_with_sum_tree(folder, round_nums=[], batch_size=batch_size, epochs=epochs)

    def raw_env_data_to_input(self, raw_data, action):
        return [raw_data[:, :, 0], raw_data[:, :, 1], raw_data[:, :, 2], raw_data[:, :, 3],
                raw_data[:, :, 4:6], raw_data[:, :, 6:8], raw_data[:, :, 8]]

    def empty_env(self):
        return [[], [], [], [], [], [], []]

    def save_model(self):
        KofAgent.save_model(self)
        self.critic.save_model()


if __name__ == '__main__':
    model = ActorCritic('iori')

    for i in range(1, 3):
        try:
            print('train ', i)
            for num in range(1, 3):
                model.train_model_with_sum_tree(i, [num], epochs=20)
                # model.train_model(i, [num], epochs=80)
                model.weight_copy()
        except:
            # print('no data in ', i)
            traceback.print_exc()
    model.save_model()

    '''

    raw_env = model.raw_data_generate(1, [1])
    train_env, train_index = model.train_env_generate(raw_env)
    train_distribution, td_error, n_action = model.actor_tarin_data(raw_env, train_env, train_index)
    # t = model.predict_model.predict([np.expand_dims(env[100], 0) for env in train_env])
    '''
