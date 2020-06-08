import os
import random
import numpy as np

from kof.kof_agent import KofAgent
from tensorflow.keras import layers
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import concatenate, BatchNormalization, CuDNNGRU, CuDNNLSTM
from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras import backend as K

from kof.value_based_models import DuelingDQN, DoubleDQN

data_dir = os.getcwd()


def PPO_loss(y_true, y_pred):
    # ratio = y_pred / y_true[:, 1]
    # return K.min(losses.categorical_crossentropy(ratio * y_true[:, 0], y_pred),
    #            losses.categorical_crossentropy(K.clip(ratio, 0.9, 1.1) * y_true[:, 0], y_pred))
    return losses.categorical_crossentropy(y_true * y_pred, y_pred)


def DDPG_loss(y_true, y_pred):
    return -K.mean(y_true)


class ActorCritic(DoubleDQN):

    def __init__(self, role, model_name='Actor'):
        self.critic = Critic(role, model_name=model_name + "_critic")
        self.train_reward_generate = self.actor_tarin_data
        super().__init__(role=role, model_name=model_name)
        self.actor = self.predict_model

    def choose_action(self, raw_data, action, random_choose=False):
        if random_choose or random.random() > self.e_greedy:
            return random.randint(0, self.action_num - 1)
        else:
            return self.predict_model.predict(self.raw_env_data_to_input(raw_data, action)).argmax()

    def build_model(self):
        shared_model = self.build_shared_model()
        t_status = shared_model.output
        output = layers.Dense(self.action_num, activation='softmax')(t_status)
        model = Model(shared_model.input, output)
        model.compile(optimizer=Adam(lr=0.00001), loss=losses.categorical_crossentropy)
        return model

    def actor_tarin_data(self, raw_env, train_env, train_index):
        _, td_error, action = self.critic.train_reward_generate(raw_env, train_env, train_index)

        # 构造onehot，将1改为td_error,
        # 使用mean(td_error * (log(action_prob)))，作为损失用于训练
        action_onehot = np.zeros(shape=(len(action[1]), self.action_num))
        action_onehot[range(len(action[1])), action[1]] = td_error
        return action_onehot, td_error, action

    def train_model_with_sum_tree(self, folder, round_nums=[], batch_size=64, epochs=30):
        # 先使用critic的td_error交叉熵训练actor，再训练critic
        # 注意这里由于PG算法限制，数据只能用一次。。。用完概率分布就改变了，公式就不满足了，PPO对此作了改进
        print('train_actor')
        # KofAgent.train_model_with_sum_tree(self, folder, round_nums=round_nums, batch_size=batch_size, epochs=1)
        KofAgent.train_model_with_sum_tree(self, folder, round_nums=round_nums, batch_size=batch_size, epochs=epochs)

        print('train_critic')
        self.critic.train_model_with_sum_tree(folder, round_nums=round_nums, batch_size=batch_size, epochs=epochs)

    def save_model(self):
        self.save_model(self)
        self.critic.save_model()

    def weight_copy(self):
        self.critic.weight_copy()
        DoubleDQN.weight_copy(self)


# PPO也是预测概率，build model
class PPO(ActorCritic):
    def __init__(self, role, model_name='PPO_actor'):
        ActorCritic.__init__(self, role=role, model_name=model_name)

    def actor_tarin_data(self, raw_env, train_env, train_index):
        _, td_error, action = self.critic.train_reward_generate(raw_env, train_env, train_index)
        # PPO用到运行时的分布
        old_prob = self.target_model.predict(train_env)[range(len(action[1])), action[1]]
        action_onehot = np.zeros(shape=(len(action[1]), self.action_num))

        # 直接将A处以采样的概率分布，keras太复杂的实现起来比较麻烦
        action_onehot[range(len(action[1])), action[1]] = td_error / old_prob
        return action_onehot, td_error, action


class DDPG(ActorCritic):
    def __init__(self, role, model_name='PPO_actor'):
        self.TAU = 0.2
        ActorCritic.__init__(self, role=role, model_name=model_name)

    def build_model(self):
        shared_model = self.build_shared_model()
        t_status = shared_model.output
        output = layers.Dense(self.action_num)(t_status)
        model = Model(shared_model.input, output)
        model.compile(optimizer=Adam(lr=0.00001), loss=PPO_loss)
        return model

    def actor_tarin_data(self, raw_env, train_env, train_index):
        _, td_error, action = self.critic.train_reward_generate(raw_env, train_env, train_index)
        # PPO用到运行时的分布
        old_prob = self.target_model.predict(train_env)[range(len(action[1])), action[1]]
        action_onehot = np.zeros(shape=(len(action[1]), self.action_num))

        # 直接将A处以采样的概率分布，keras太复杂的实现起来比较麻烦
        action_onehot[range(len(action[1])), action[1]] = td_error / old_prob
        return [-action_onehot, td_error, action]

    def weight_copy(self):
        """soft update target model.
        formula：θ​​t ← τ * θ + (1−τ) * θt, τ << 1.
        """
        critic_weights = self.critic.predict_model.get_weights()
        critic_target_weights = self.critic.target_model.get_weights()

        actor_weights = self.predict_model.get_weights()
        actor_target_weights = self.target_model.get_weights()

        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]

        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]

        self.critic.target_model.set_weights(critic_target_weights)
        self.target_model.set_weights(actor_target_weights)


class Critic(DoubleDQN):
    def __init__(self, role, model_name='critic'):
        DoubleDQN.__init__(self, role=role, model_name=model_name)
        self.train_reward_generate = self.critic_dqn_train_data

    def build_model(self):
        shared_model = self.build_shared_model()
        t_status = shared_model.output
        t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
        t_status = BatchNormalization()(t_status)
        t_status = layers.LeakyReLU(0.05)(t_status)

        # critic只输出和状态有关的v值
        value = layers.Dense(1)(t_status)

        model = Model(shared_model.input, value)

        model.compile(optimizer=Adam(lr=0.00001), loss='mse')

        return model

    # 只有状态价值v值，标量
    def critic_dqn_train_data(self, raw_env, train_env, train_index):
        reward = raw_env['reward'].reindex(train_index)
        action = raw_env['action'].reindex(train_index)
        action = action.astype('int')

        target_model_prediction = self.target_model.predict(train_env)
        predict_model_prediction = self.predict_model.predict(train_env)
        # pre_prediction = predict_model_prediction.copy()

        time = raw_env['time'].reindex(train_index).diff(1).shift(-1).fillna(1).values

        next_action_reward = target_model_prediction[range(len(train_index))]

        next_action_reward = np.roll(next_action_reward, -1)
        next_action_reward[time > 0] = 0

        reward = np.expand_dims(reward, 1)
        reward += self.reward_decay * next_action_reward

        td_error = reward - predict_model_prediction[range(len(train_index))]
        td_error = td_error.flatten()
        predict_model_prediction[range(len(train_index))] = reward

        return [predict_model_prediction, td_error, [None, action]]


if __name__ == '__main__':
    # model1 = ActorCritic('iori')
    # model2 = Critic('iori')
    model2 = PPO('iori')
    model3 = DDPG('iori')
    # model3.train_model_with_sum_tree(0, [0], epochs=40)
    '''
    for i in range(1, 10):
        try:
            print('train ', i)
            for num in range(1, 3):
                model.train_model_with_sum_tree(i, [num], epochs=40)
                # model.train_model(i, [num], epochs=80)
                model.weight_copy()
        except:
            # print('no data in ', i)
            traceback.print_exc()
    model.save_model()



    raw_env = model.raw_data_generate(1, [1])
    train_env, train_index = model.train_env_generate(raw_env)
    train_distribution, td_error, n_action = model.actor_tarin_data(raw_env, train_env, train_index)
    # t = model.predict_model.predict([np.expand_dims(env[100], 0) for env in train_env])
    '''
