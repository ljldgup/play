import os
import traceback
import numpy as np
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from kof.kof_agent import KofAgent
from tensorflow.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizers import Adam

from kof.shared_model import build_multi_attention_model

'''
可调参数
输入
网络结构：距离,坐标先卷积再输入rnn效果更好
衰减因子:不能太大，或者太小，0.95左右
reward比例:不能太小，不然难以收敛
输入步数
学习率 过大的话学习效果很不好，小一点容易找到稳定的策略

改进点：
残差网络
transformer

减小过估计的几个注意点
经过一定计算后再更新target模型，不是每次训练后都加入，这点最重要
reward注意不要计入多余的值, reward的范围不能太大，大约在大约在0.5左右
'''

data_dir = os.getcwd()


# 普通DDQN，输出分开成多个fc，再合并
# 这种方法违背网络共享信息的特点
# 位置距离卷积 + lstm + fc
# 接近BN层貌似一定程度上会导致过估计，所以暂时删掉
class DoubleDQN(KofAgent):

    def __init__(self, role, model_name='double_dqn', reward_decay=0.94):
        super().__init__(role=role, model_name=model_name, reward_decay=reward_decay)
        # 把target_model移到value based文件中,因为policy based不需要
        self.target_model = self.build_model()
        self.target_model.set_weights(self.predict_model.get_weights())
        self.train_reward_generate = self.double_dqn_train_data

    def build_model(self):
        # shared_model = self.build_shared_model()
        shared_model = build_multi_attention_model(self.input_steps)
        t_status = shared_model.output
        output = layers.Dense(self.action_num, kernel_initializer='he_uniform')(t_status)
        model = Model(shared_model.input, output, name=self.model_name)

        model.compile(optimizer=Adam(lr=0.00001), loss='mse')

        return model

    # 每次训练完就直接跟新target的参数就是nature dqn
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
        next_q = target_model_prediction[range(len(train_index)), next_max_reward_action.astype('int')]
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
        td_error = td_error.values

        # 这里报action过多很可能是人物不对
        predict_model_prediction[range(len(train_index)), action] = reward

        # 上下限裁剪，防止过估计
        predict_model_prediction[predict_model_prediction > 2] = 2
        predict_model_prediction[predict_model_prediction < -2] = -2
        return [predict_model_prediction, td_error, [pre_actions, action.values]]

    def train_model(self, folder, round_nums=[], batch_size=64, epochs=30):
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
        KofAgent.save_model(self)

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
        ax1 = fig1.add_subplot(111)
        ax1.hist(ans.flatten(), bins=30, label=self.model_name)
        fig1.legend()


# 正常dueling dqn
# 将衰减降低至0.94，去掉了上次动作输入，将1p embedding带宽扩展到8，后效果比之前好了很多
# 但动作比较集中
class DuelingDQN(DoubleDQN):
    def __init__(self, role, model_name='dueling_dqn', reward_decay=0.94):
        super().__init__(role=role, model_name=model_name, reward_decay=reward_decay)

    def build_model(self):
        # shared_model = self.build_shared_model()
        shared_model = build_multi_attention_model(self.input_steps)
        t_status = shared_model.output

        # 攻击动作，则采用基础标量 + 均值为0的向量策略
        value = layers.Dense(1)(t_status)
        # 这力可以直接广播，不需要拼接
        # value = concatenate([value] * self.action_num)
        a = layers.Dense(self.action_num)(t_status)
        mean = layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantage = layers.Subtract()([a, mean])
        q = layers.Add()([value, advantage])
        model = Model(shared_model.input, q, name=self.model_name)

        model.compile(optimizer=Adam(lr=0.000001), loss='mse')

        return model


def train_model(model, folders):
    print('-----------------------------')
    print('train ', model.model_name)
    # 在刚开始训练网络的时候使用
    model.multi_steps = 2
    for i in folders:
        try:
            print('train ', i)
            # model.train_model(i)
            model.train_model(i, epochs=20)
            # 这种直接拷贝的效果和nature DQN其实没有区别。。所以放到外层去拷贝，训练时应该加大拷贝的间隔
            # 改成soft copy

        except:
            traceback.print_exc()
        if i % 3:
            model.weight_copy()


if __name__ == '__main__':
    # models = [DuelingDQN('iori'), DoubleDQN('iori')]
    model = DuelingDQN('iori')

    # model.model_test(1, [1,2])
    # model.model_test(2, [1,2])
    # model.predict_model.summary()
    # t = model.operation_analysis(5)
    # model.train_model(5, epochs=40)
    train_model(model, range(1,10))
    model.weight_copy()
    model.save_model()
    model.value_test(15, [1])

    '''
    raw_env = model.raw_data_generate(1, [1])
    train_env, train_index = model.train_env_generate(raw_env)
    train_reward, td_error, n_action = model.double_dqn_train_data(raw_env, train_env, train_index)
    # 这里100 对应的是 raw_env 中 100+input_steps左右位置
    t = model.predict_model.predict([np.expand_dims(env[100], 0) for env in train_env])
    # output = model.output_test([ev[50].reshape(1, *ev[50].shape) for ev in train_env])
    # train_reward[range(len(n_action[1])), n_action[1]]
    # model.model_test(1, [1])


    # 查看训练数据是否对的上
    index = 65
    train_index[index], raw_env['action'].reindex(train_index).values[index], raw_env['reward'].reindex(
        train_index).values[index], [np.expand_dims(env[index], 0) for env in train_env]
    '''
