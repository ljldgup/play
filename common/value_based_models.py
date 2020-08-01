import os
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from common.agent import CommonAgent
from tensorflow.keras import layers
from tensorflow.python.keras import Model
from tensorflow.keras.optimizers import Adam

'''
可调的点

网络结构：距离,坐标先卷积再输入rnn效果更好，使用堆叠rnn，rnn+attention, transformer 效果不同
衰减因子:不能太大，或者太小，0.95左右
reward比例:不能太小或者太大都容易过估计，reward 从30 扩大到60，效果很明显变差20-10效果也变差
输入步数：输入步数从10减小到6之后有明显的进步，6到4效果又变差
学习率： 过大的话学习效果很不好，小一点容易找到稳定的策略，可以根据前后策略变化情况来看
输出空间:不宜过大，不然很难找到稳定的策略
拷贝间隔:不宜过小，否则容易过估计
多步学习：
目前测出来比较有效的，rnn堆叠，衰减因子0.99，缩放比例30-20，输入步数6，学习率1e-6 - 1e-5，输出空间<15
改进点：
可以考虑把采样过程加入到网络结果，把预测和训练模型分开，这样可以加快即时速度

减小过估计的几个注意点
经过一定计算后再更新target模型，不是每次训练后都加入，这点最重要
最后几层使用bn层也会造成很明显的过估计，目前加一层，不加几乎无法收敛,怀疑网络是否真的有效？？
去掉输出位置的bn层以后收敛效果好了一些，小批量数据用bn层效果貌似不好
reward注意不要计入多余的值, reward的范围不能太大太小，在±0.1-1左右

训练时可以对训练的值进行裁剪，裁剪区间对模型训练也有影响，区间太小，减小溢出值将成为网络主要梯度来源

'''

data_dir = os.getcwd()


# 自定义损失，无关动作不提供梯度
def dqn_loss(y_true, y_pred):
    sign = tf.where(y_true == 0., 0., 1.)
    delta = y_true - y_pred
    loss = tf.reduce_mean(delta * delta * sign)
    return loss


# 普通DDQN，输出分开成多个fc，再合并
# 这种方法违背网络共享信息的特点
# 位置距离卷积 + lstm + fc
# 接近BN层貌似一定程度上会导致过估计，所以暂时删掉
class DoubleDQN(CommonAgent):

    def __init__(self, role, action_num, functions, model_type='double_dqn'):
        super().__init__(role=role, action_num=action_num, functions=functions,
                         model_type=model_type)
        # 把target_model移到value based文件中,因为policy based不需要
        self.train_reward_generate = self.double_dqn_train_data

    def build_model(self):
        shared_model = self.base_network_build_fn()
        # shared_model = build_multi_attention_model(self.input_steps)
        t_status = shared_model.output
        output = layers.Dense(self.action_num, kernel_initializer='he_uniform')(t_status)
        model = Model(shared_model.input, output, name=self.model_type)

        model.compile(optimizer=Adam(lr=self.lr), loss=dqn_loss)

        return model

    # 每次训练完就直接跟新target的参数就是nature dqn
    def double_dqn_train_data(self, raw_env, train_env, train_index):
        reward = raw_env['reward'].reindex(train_index).values
        action = raw_env['action'].reindex(train_index).values
        action = action.astype('int')

        target_model_prediction = self.target_model.predict(train_env)
        predict_model_prediction = self.predict_model.predict(train_env)
        # pre_prediction = predict_model_prediction.copy()
        pre_actions = predict_model_prediction.argmax(axis=1)

        # 由训练模型选动作，target模型根据动作估算q值，不关心是否最大
        # yj=Rj + γQ′(ϕ(S′j), argmaxa′Q(ϕ(S′j), a, w), w′)
        time = raw_env['time'].reindex(train_index).diff(1).shift(-1).fillna(1).values
        # 这里应该用采取的行为，如果采用过去的记录训练，这里预测值不是实际采取的值
        next_q = target_model_prediction[range(len(train_index)), action]
        '''
        print(predict_model_prediction[range(20), action[:20]])
        print(reward.values[:20])
        print(next_q[:20])
        reward += self.reward_decay * next_q
        '''
        # multi-Step Learning 一次性加上后面n步的实际报酬 在加上n+1目标网络的q值，
        # multi_steps适合再训练初期网络与实际偏差较大的情况下使用
        # 这样能加速训练，但会导致网络无法找到稳定的获利策略，训练一段时间后应该将multi_steps改成1

        print('multi steps: ', self.multi_steps)
        # 将1到multi_steps个步骤的r，乘以衰减后相加
        # 这里原本有一个所以只需要做multi_steps - 1 次
        next_reward = reward.copy()
        for i in range(self.multi_steps - 1):
            next_reward = np.roll(next_reward, -1)
            # 从下一个round往上移动的，reward为0
            next_reward[time > 0] = 0
            reward += next_reward * self.reward_decay ** (i + 1)

        # 加上target model第multi_steps+1个步骤的Q值
        for i in range(self.multi_steps):
            next_q = np.roll(next_q, -1)
            # 从下一个round往上移动的，reward为0
            next_q[time > 0] = 0
        reward += next_q * self.reward_decay ** self.multi_steps

        td_error = reward - predict_model_prediction[range(len(train_index)), action]
        # 这里报action过多很可能是人物不对
        train_reward = np.zeros_like(predict_model_prediction)
        train_reward[range(len(train_index)), action] = reward
        '''
        # 上下限裁剪，防止过估计
        predict_model_prediction[predict_model_prediction > 4] = 4
        predict_model_prediction[predict_model_prediction < -4] = -4
        '''
        return [train_reward, td_error, [pre_actions, action]]

    def train_model(self, folder, round_nums=[], batch_size=16, epochs=30):
        self.train_model_with_sum_tree(folder, round_nums, batch_size, epochs)

    # soft型,指数平滑拷贝
    def soft_weight_copy(self):
        mu = 0.05
        weights = self.predict_model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(weights)):
            target_weights[i] = mu * weights[i] + (1 - mu) * target_weights[i]

        self.target_model.set_weights(target_weights)

    def weight_copy(self):
        self.target_model.set_weights(self.predict_model.get_weights())

    def save_model(self, ):
        self.weight_copy()
        CommonAgent.save_model(self)

    def value_test(self, folder, round_nums):
        # q值分布可视化
        raw_env = self.raw_env_generate(folder, round_nums)
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
        ax1 = fig1.add_subplot(111)
        ax1.hist(ans.flatten(), bins=30, label=self.model_type)
        fig1.legend()

        fig2 = plt.figure()
        for i in range(self.action_num):
            ax1 = fig2.add_subplot(4, 4, i + 1)
            ax1.hist(ans[:, i], bins=20)
            fig2.legend()

        self.critic.value_test(folder, round_nums)


# 正常dueling dqn
# 将衰减降低至0.94，去掉了上次动作输入，将1p embedding带宽扩展到8，后效果比之前好了很多
# 但动作比较集中
class DuelingDQN(DoubleDQN):
    def __init__(self, role, action_num, functions, model_type='dueling_dqn', ):
        super().__init__(role=role, action_num=action_num, functions=functions,
                         model_type=model_type)

    def build_model(self):
        # shared_model = build_stacked_rnn_model(self)
        shared_model = self.base_network_build_fn()

        # shared_model = build_multi_attention_model(self)
        t_status = shared_model.output

        # 攻击动作，则采用基础标量 + 均值为0的向量策略
        value = layers.Dense(1)(t_status)
        # 这力可以直接广播，不需要拼接
        #         # value = concatenate([value] * self.action_num)
        a = layers.Dense(self.action_num)(t_status)
        mean = layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = layers.Subtract()([a, mean])
        q = layers.Add()([value, advantage])
        model = Model(shared_model.input, q, name=self.model_type)

        model.compile(optimizer=Adam(lr=self.lr), loss='mse')

        return model


if __name__ == '__main__':
    pass
