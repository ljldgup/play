import os
import random
import traceback

import numpy as np

from kof.kof_agent import KofAgent, train_model_1by1, get_maxsize_file
from tensorflow.keras import layers
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras import backend as K
import tensorflow as tf
from matplotlib import pyplot as plt

from kof.shared_model import build_multi_attention_model
from kof.value_based_models import DoubleDQN

tf.compat.v1.disable_eager_execution()
data_dir = os.getcwd()
epsilon = 0.01


def DDPG_loss(y_true, y_pred):
    # DDPG返回的梯度应该是-R,这里通过相乘实现，不知道对不对
    return -K.mean(y_true * y_pred)


# 这里发现一个问题，就是使用model.fit进行训练，无论返回正还是负的，最终训练都是用正的
# 因为 有可能是负的，所以loss有可能是负数
class PPO_Loss(layers.Layer):
    def __init__(self, **kwargs):
        super(PPO_Loss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        Layer的call方法明确要求inputs为一个tensor，或者包含多个tensor的列表/元组
        所以这里不能直接接受多个入参，需要把多个入参封装成列表/元组的形式然后在函数中自行解包，否则会报错。
        """
        # 解包入参
        r, p_old, p_new = inputs

        # 有时概率会为0，导致输出loss极大，做剪裁来避免
        p_new = K.clip(p_new, 0.001, 1)

        # 复杂的损失函数
        ration = p_new / p_old
        adv = r * K.log(p_new)
        loss = -K.mean(K.minimum(ration * adv,
                                 K.clip(ration, 1. - epsilon, 1. + epsilon) * adv))
        self.add_loss(-K.mean(ration), inputs=inputs)
        return loss


class ActorCritic(DoubleDQN):

    def __init__(self, role, model_name='AC_actor'):
        self.critic = Critic(role, model_name=model_name + "_critic")
        super().__init__(role=role, model_name=model_name)
        self.train_reward_generate = self.actor_tarin_data
        self.actor = self.predict_model

    def choose_action(self, raw_data, random_choose=False):
        if random_choose or random.random() > self.e_greedy:
            return random.randint(0, self.action_num - 1)
        else:
            # 使用np.random.choice返回采样结果
            prob = self.predict_model.predict(self.raw_env_data_to_input(raw_data))
            return np.random.choice(self.action_num, p=prob[0])

    def build_model(self):
        # shared_model = self.build_shared_model()
        shared_model = build_multi_attention_model(self.input_steps)
        t_status = shared_model.output
        output = layers.Dense(self.action_num, activation='softmax')(t_status)
        model = Model(shared_model.input, output, name=self.model_name)
        model.compile(optimizer=Adam(lr=0.00001), loss=losses.categorical_crossentropy)
        return model

    def actor_tarin_data(self, raw_env, train_env, train_index):
        adv, td_error, action = self.critic.train_reward_generate(raw_env, train_env, train_index)
        # 构造onehot，将1改为td_error,
        # 使用mean(td_error * (log(action_prob)))，作为损失用于训练

        # 使用一定的噪声初始化，尽可能避免概率收敛到过激值
        action_onehot = np.random.rand(len(action[1]), self.action_num) / 100

        # 这里即可使用价值，也可以同td_error
        action_onehot[range(len(action[1])), action[1]] = td_error
        return action_onehot, td_error, action

    def train_model(self, folder, round_nums=[], batch_size=64, epochs=30):
        # 先使用critic的td_error交叉熵训练actor，再训练critic
        # 注意这里由于PG算法限制，数据只能用一次。。。用完概率分布就改变了，公式就不满足了，PPO对此作了改进
        print(self.model_name)
        # KofAgent.train_model_with_sum_tree(self, folder, round_nums=round_nums, batch_size=batch_size, epochs=1)
        KofAgent.train_model(self, folder, round_nums, batch_size, epochs)
        print('train_critic')
        self.critic.train_model(folder, round_nums=round_nums, batch_size=batch_size, epochs=epochs)

    def save_model(self):
        DoubleDQN.save_model(self)
        self.critic.save_model()

    def soft_weight_copy(self):
        self.critic.soft_weight_copy()
        DoubleDQN.soft_weight_copy()

    def weight_copy(self):
        self.critic.weight_copy()
        DoubleDQN.weight_copy(self)


# PPO也是预测概率，build model
class PPO(ActorCritic):

    def __init__(self, role, model_name='PPO_actor'):
        ActorCritic.__init__(self, role=role, model_name=model_name)

        # 用于训练的模型
        self.trained_model = self.build_train_model()
        self.trained_model.set_weights(self.predict_model.get_weights())
        self.copy_interval = 6

    def build_model(self):
        # shared_model = build_attention_model(self.input_steps, self.action_num)
        shared_model = build_multi_attention_model(self.input_steps)
        t_status = shared_model.output
        output = layers.Dense(self.action_num, activation='softmax')(t_status)
        model = Model(shared_model.input, output, name=self.model_name)
        # 这里的优化器，损失都没用
        model.compile(optimizer=Adam(lr=0.00001), loss='mse')
        return model

    # 这个模型加了损失层PPO_Loss，用于训练
    def build_train_model(self):
        # shared_model = build_attention_model(self.input_steps, self.action_num)
        shared_model = build_multi_attention_model(self.input_steps)
        t_status = shared_model.output
        output = layers.Dense(self.action_num, activation='softmax')(t_status)

        r = Input(shape=(self.action_num,), name='r')
        old_prob = Input(shape=(self.action_num,), name='old_prob')

        loss = PPO_Loss()([r, old_prob, output])
        input = shared_model.input + [r, old_prob]
        model = Model(input, loss, name=self.model_name)
        # 减小学习率，避免策略遗忘的太快
        model.compile(optimizer=Adam(lr=0.000001), loss=None)
        return model

    def actor_tarin_data(self, raw_env, train_env, train_index):
        adv, td_error, action = self.critic.train_reward_generate(raw_env, train_env, train_index)
        # PPO用到运行时的分布
        old_prob = self.target_model.predict(train_env)
        old_action = old_prob.argmax(axis=1)
        # 即使softmax也有可能是0,或者极小，这样就导致计算得到nan,或者非常大，
        # 这里设置成一个相对小的数
        old_prob[old_prob < 0.001] = 0.001
        reward_onehot = np.random.rand(len(action[1]), self.action_num) / 1000 + 0.001
        # 这里用adv或者te_error都可以
        # reward_onehot[range(len(action[1])), action[1]] = adv[:, 0]
        reward_onehot[range(len(action[1])), action[1]] = td_error
        return [[reward_onehot, old_prob], td_error, [old_action, action[1]]]

    # 暂时不用sumtree
    def train_model(self, folder, round_nums=[], batch_size=64, epochs=30):
        # 先使用critic的td_error交叉熵训练actor，再训练critic
        # 注意这里由于PG算法限制，数据只能用一次。。。用完概率分布就改变了，公式就不满足了，PPO对此作了改进
        print('train ', self.model_name)

        if not round_nums:
            round_nums = get_maxsize_file(folder)

        raw_env = self.raw_data_generate(folder, round_nums)
        train_env, train_index = self.train_env_generate(raw_env)
        train_target, td_error, action = self.train_reward_generate(raw_env, train_env, train_index)

        # PPO的损失比较复杂放在最后一个损失层来实现，reward等一起作为输入
        train_env += train_target

        self.trained_model.set_weights(self.predict_model.get_weights())

        loss_history = []
        print('train {}/{} {} epochs'.format(folder, round_nums, epochs))
        for e in range(epochs):
            loss = 0
            for i in range(len(train_index) // batch_size):
                index = range(i * batch_size, (i + 1) * batch_size)
                # train_model loss层没有y
                loss += self.trained_model.train_on_batch([env[index] for env in train_env])
            print(loss / (len(train_index) // batch_size))
            loss_history.append(loss)

        self.record['total_epochs'] += epochs
        # PPO的参数每轮都需要更新一次，概率分布不能太远
        self.predict_model.set_weights(self.trained_model.get_weights())
        self.target_model.set_weights(self.trained_model.get_weights())
        print('train_critic')
        self.critic.train_model(folder, round_nums=round_nums, batch_size=batch_size, epochs=epochs)

    def value_test(self, folder, round_nums):
        # q值分布可视化
        raw_env = self.raw_data_generate(folder, round_nums)
        train_env, train_index = self.train_env_generate(raw_env)
        ans = self.predict_model.predict(train_env)
        # 这里是训练数据但是也可以拿来参考,查看是否过估计，目前所有的模型几乎都会过估计
        print('max reward:', ans.max())
        print('min reward:', ans.min())
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
            ax1.hist(ans[:, i], bins=20)
            fig1.legend()

        self.critic.value_test(folder, round_nums)


# DDPG适合连续的动作空间，用在这里不合适
class DDPG(ActorCritic):
    def __init__(self, role, model_name='PPO_actor'):
        self.TAU = 0.2
        ActorCritic.__init__(self, role=role, model_name=model_name)
        self.copy_interval = 1

    def build_model(self):
        shared_model = self.build_shared_model()
        t_status = shared_model.output
        output = layers.Dense(self.action_num)(t_status)
        model = Model(shared_model.input, output, name=self.model_name)
        # 这里的优化器，损失没有用
        model.compile(optimizer=Adam(lr=0.00001), loss=DDPG_loss)
        return model

    def actor_tarin_data(self, raw_env, train_env, train_index):
        _, td_error, action = self.critic.train_reward_generate(raw_env, train_env, train_index)
        # PPO用到运行时的分布
        action_onehot = np.random.rand(10, 20)[len(action[1]), self.action_num] / 100

        action_onehot[range(len(action[1])), action[1]] = td_error
        return [action_onehot, td_error, action]


class Critic(DoubleDQN):
    def __init__(self, role, model_name='critic'):
        # policy gradient 应该是个连续的情况，所以这里用比较高的reward_decay
        DoubleDQN.__init__(self, role=role, model_name=model_name, reward_decay=0.94)
        self.train_reward_generate = self.critic_dqn_train_data
        # 这设置一个较大multi_steps值
        self.multi_steps = 2

    def build_model(self):
        # shared_model = self.build_shared_model()
        shared_model = build_multi_attention_model(self.input_steps)
        t_status = shared_model.output
        t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
        t_status = BatchNormalization()(t_status)
        t_status = layers.LeakyReLU(0.05)(t_status)

        # critic只输出和状态有关的v值
        value = layers.Dense(1)(t_status)

        model = Model(shared_model.input, value)

        model.compile(optimizer=Adam(lr=0.000001), loss='mse')

        return model

    # 只有状态价值v值，标量
    def critic_dqn_train_data(self, raw_env, train_env, train_index):
        reward = raw_env['reward'].reindex(train_index)
        reward = reward.values
        action = raw_env['action'].reindex(train_index)
        action = action.astype('int')

        target_model_prediction = self.target_model.predict(train_env)
        predict_model_prediction = self.predict_model.predict(train_env)
        # pre_prediction = predict_model_prediction.copy()

        time = raw_env['time'].reindex(train_index).diff(1).shift(-1).fillna(1).values

        print('multi steps: ', self.multi_steps)
        # , 0是为了reward相加结构对称
        next_q = target_model_prediction[range(len(train_index)), 0]
        next_reward = reward.copy()
        for i in range(self.multi_steps - 1):
            next_reward = np.roll(next_reward, -1)
            # 从下一个round往上移动的，reward为0
            next_reward[time > 0] = 0
            reward += next_reward
            # reward += next_reward * self.reward_decay ** (i + 1)
            # 加上target model第multi_steps+1个步骤的Q值
        for i in range(self.multi_steps):
            next_q = np.roll(next_q, -1)
            # 从下一个round往上移动的，reward为0
            next_q[time > 0] = 0
        reward += next_q * self.reward_decay ** self.multi_steps

        td_error = reward - predict_model_prediction[range(len(train_index)), 0]
        predict_model_prediction[range(len(train_index)), 0] = reward

        return [predict_model_prediction, td_error, [None, action]]


if __name__ == '__main__':
    # model1 = ActorCritic('iori')
    # model2 = Critic('iori')

    model2 = PPO('iori')

    # model2.train_model(15, [1])
    # model3 = DDPG('iori')

    train_model_1by1(model2, range(1, 11), range(1, 10))
    model2.value_test(2, [1])
    model2.save_model()
    '''
    model2.model_test(5, [1])
    '''
    '''
    raw_env = model2.raw_data_generate(1, [1])
    train_env, train_index = model2.train_env_generate(raw_env)
    train_distribution, td_error, n_action = model2.actor_tarin_data(raw_env, train_env, train_index)
    t = model2.predict_model.predict([np.expand_dims(env[100], 0) for env in train_env])
    t = model2.predict_model.predict(train_env)
    # 这里发现loss为nan，因为损失已经上溢，某些p过小，导致做除数或者对数返回值过大，目前都做了剪裁
    # np.sum(train_distribution[0] * t / train_distribution[1] * np.log(t))
    '''
