import os
import random
import time
import traceback
import types

import numpy as np
from tensorflow.python.keras import Model
from kof.kof_command_mame import role_commands
from common.sumtree import SumTree


class RandomAgent:
    def __init__(self, **kwargs):
        self.model_name = 'random'
        self.role = kwargs['role']
        self.action_num = kwargs['action_num']
        self.input_steps = 1
        # 操作间隔步数
        self.operation_interval = 2

    def choose_action(self, *args, **kwargs):
        return random.randint(0, self.action_num - 1)

    def save_model(self):
        pass


data_dir = os.getcwd()


# 选择效果比较好，即文件比较大的记录
def get_maxsize_file(folder):
    files = os.listdir('{}/{}'.format(data_dir, folder))
    data_files = filter(lambda f: '.' in f, files)
    round_nums = np.array(list(set([file.split('.')[0] for file in data_files])))
    file_size = map(lambda num: os.path.getsize('{}/{}/{}.env'.format(data_dir, folder, num)), round_nums)
    # 取前6个
    return round_nums[np.array(list(file_size)).argsort()].tolist()[-6:]


def train_model_1by1(model, folders, rounds):
    print('-----------------------------')
    print('train ', model.model_type)

    # 在刚开始训练网络的时候使用
    # model.multi_steps = 6
    count = 0
    for i in folders:
        for r in rounds:
            try:
                print('train {}/{}'.format(i, r))
                # model.train_model(i)
                model.train_model(i, [r], epochs=30)
                # 这种直接拷贝的效果和nature DQN其实没有区别。。所以放到外层去拷贝，训练时应该加大拷贝的间隔
                # 改成soft copy
                # model.soft_weight_copy()
                count += 1
            except:
                traceback.print_exc()
            if count % model.copy_interval == 0:
                model.weight_copy()
    model.save_model()


# 这里通过继承修改raw_env_generate，train_env_generate,empty_env，raw_env_data_to_input改变环境对应对应生成方式
# 以及build_model中共享的模型函数，可以把KofAgent调整成为其他需要的Agent
# 创建base_network_build_fn部分在初始化时就会用
# 保存在具体操作时进行，每个回合保存一次，根据raw_env_generate的保存方法进行
class CommonAgent:
    def __init__(self, role, action_num, functions, model_type):
        self.role = role
        self.model_type = model_type
        self.action_num = action_num
        # base_network_build_fn 基础模型生成函数，用来调整rnn attention，transformer等公共部分生成
        # 必须绑定，否则无法传入self
        self.base_network_build_fn = types.MethodType(functions[0], self)
        self.raw_env_generate = types.MethodType(functions[1], self)
        self.train_env_generate = types.MethodType(functions[2], self)
        self.raw_env_data_to_input = types.MethodType(functions[3], self)
        self.empty_env = types.MethodType(functions[4], self)
        self.reward_decay = 0.999
        # 输入步数
        self.input_steps = 6

        # 训练时会重置
        self.e_greedy = 0

        # 操作间隔时间步数
        self.operation_interval = 2
        # 由于action有间隔，输入序列第一个action所在的位置，方便提取action
        self.action_begin_index = (self.input_steps - 1) % self.operation_interval
        # 小于1在transformer中会出错
        self.action_steps = (self.input_steps - 1) // self.operation_interval

        # multi_steps 配合decay使网络趋向真实数据，但multi_steps加大会导致r波动大
        self.multi_steps = 1

        # 模型参数拷贝间隔
        self.copy_interval = 5
        # reward 缩减比例
        self.reward_scale_factor = 20
        # 学习率
        self.lr = 2e-6
        # build_model由子类提供
        self.predict_model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.predict_model.get_weights())
        self.record = self.get_record()
        self.load_model()

    def choose_action(self, raw_data, action, random_choose=False):
        if random_choose or random.random() > self.e_greedy:
            return random.randint(0, self.action_num - 1)
        else:
            # 根据dqn的理论，这里应该用训练的模型， target只在寻训练的时候使用，并且一定时间后才复制同步
            return self.predict_model.predict(self.raw_env_data_to_input(raw_data, action)).argmax()

    # 读入数据记录文件，并计算reward
    def raw_env_generate(self, folder, round_nums):
        '''
        if round_nums:
            if os.path.exists('{}/{}/{}.env'.format(data_dir, folder, round_nums[0])):
                raw_env = np.loadtxt('{}/{}/{}.env'.format(data_dir, folder, round_nums[0]))
                raw_actions = np.loadtxt('{}/{}/{}.act'.format(data_dir, folder, round_nums[0]))
            else:
                raise Exception('no {}/{}/{}.env'.format(data_dir, folder, round_nums[0]))
            for i in range(1, len(round_nums)):
                if os.path.exists('{}/{}/{}.env'.format(data_dir, folder, round_nums[i])):
                    # print(i)
                    raw_env = np.concatenate([raw_env,
                                              np.loadtxt('{}/{}/{}.env'.format(data_dir, folder, round_nums[i]))])
                    raw_actions = np.concatenate([raw_actions,
                                                  np.loadtxt(
                                                      '{}/{}/{}.act'.format(data_dir, folder, round_nums[i]))])

            raw_env = pd.DataFrame(raw_env, columns=[])
            raw_env['action'] = raw_actions
            raw_env['reward'] = ....
            return raw_env
        '''

        pass

    # 生成未加衰减奖励的训练数据
    # 这里把文件夹换成env_reward_generate生成的数据，避免过度耦合
    def train_env_generate(self, raw_env):
        # return [train_env_data, train_index]
        pass

    # 生成基础模型
    def build_base_network(self):
        pass

    # 最开始用的不考虑长期收益的模型，在pandas钟使用rolling来使其拥有少量的长期效果
    # 现在使用该函数实现多态
    def train_reward_generate(self, raw_env, train_env, train_index):
        return [None, None, [None, None]]

    # 在线学习可以batch_size设置成
    # 改成所有数据何在一起，打乱顺序，使用batch 训练，速度快了很多
    # batch_size的选取不同，损失表现完全不一样
    def train_model(self, folder, round_nums=[], batch_size=16, epochs=30):
        if not round_nums:
            round_nums = get_maxsize_file(folder)

        raw_env = self.raw_env_generate(folder, round_nums)
        train_env, train_index = self.train_env_generate(raw_env)
        train_target, td_error, action = self.train_reward_generate(raw_env, train_env, train_index)
        # 感觉强化学习打乱数据的意义不大
        # random_index = np.random.permutation(len(train_index))
        # verbose参数控制输出，这里每个epochs输出一次
        history = self.predict_model.fit(train_env, train_target,
                                         batch_size=batch_size,
                                         epochs=epochs, verbose=2)

        self.record['total_epochs'] += epochs
        return history

    # Prioritized Replay DQN 使用sumtree 生成 batch
    # 由于sumtree是根据td_error随机采样，每个batch或者epochs都重新采样，返回的loss波动很大，几乎不收敛
    # 将采样改成batch为基础单元，改成所有训练均用同一的采样
    def train_model_with_sum_tree(self, folder, round_nums=[], batch_size=32, epochs=30):
        if not round_nums:
            round_nums = get_maxsize_file(folder)
        # raw_env_generate,train_env_generate 具体的agent实现
        raw_env = self.raw_env_generate(folder, round_nums)
        train_env, train_index = self.train_env_generate(raw_env)
        train_target, td_error, action = self.train_reward_generate(raw_env, train_env, train_index)

        # 根据batch内td error绝对值和生成sumtree

        print('train with sum tree {}/{} {} epochs'.format(folder, round_nums, epochs))
        sum_tree = SumTree(abs(td_error))
        index = sum_tree.get_index(len(td_error))
        history = self.predict_model.fit([env[index] for env in train_env], train_target[index],
                                         batch_size=batch_size,
                                         epochs=epochs, verbose=2)
        self.record['total_epochs'] += epochs
        '''
        # 使用batch的td_error生成sumtree，容易引起cudnnlstm层出错，原因不明。。有可能使batch_size太小
        batch_sum_tree = self.get_batch_sumtree(td_error, batch_size)
        batch_index = batch_sum_tree.gen_batch_index(len(td_error) // batch_size)
        for e in range(epochs):
            loss = 0
            for idx in batch_index:
                loss += self.predict_model.train_on_batch(
                    [env[idx * batch_size:idx * batch_size + batch_size] for env in train_env],
                    train_target[idx * batch_size:idx * batch_size + batch_size])
            loss_history.append(loss)
            print(loss / (len(train_index) // batch_size))
        '''
        '''
        # 测试训练完后概率变化程度
        new_r = self.predict_model.predict([env for env in train_env])
        for o, n, td in zip(old_r[range(len(action[1])), action[1]],
                                     new_r[range(len(action[1])), action[1]],
                                     td_error):
            print(o, n, td)
        '''
        return history

    # 根据batch内td error绝对值和生成sumtree
    # td_error 形状num,error
    def get_batch_sumtree(self, td_error, batch_size):
        batch_num = len(td_error) // batch_size
        batch_td_error = td_error[:batch_num * batch_size].reshape(batch_num, batch_size, -1)

        batch_td_error = abs(batch_td_error)
        batch_td_error = batch_td_error.sum(axis=1).flatten()
        batch_td_error = np.r_[batch_td_error, abs(td_error[batch_num * batch_size:]).sum()]
        return SumTree(batch_td_error)

    def save_model(self):
        print('save {}/model/{}_{}_{}'.format(data_dir, self.role, self.model_type, self.network_type))
        if not os.path.exists('{}/model'.format(data_dir)):
            os.mkdir('{}/model'.format(data_dir))
        self.predict_model.save_weights(
            '{}/model/{}_{}_{}'.format(data_dir, self.role, self.model_type, self.network_type))

        with open('{}/model/{}_{}_{}_record'.format(data_dir, self.role, self.model_type, self.network_type), 'w') as r:
            r.write(str(self.record))

    def load_model(self):
        if os.path.exists('{}/model/{}_{}_{}_record'.format(data_dir, self.role, self.model_type, self.network_type)):
            print('load {}/model/{}_{}_{}'.format(data_dir, self.role, self.model_type, self.network_type))
            self.predict_model.load_weights(
                '{}/model/{}_{}_{}'.format(data_dir, self.role, self.model_type, self.network_type))
        else:
            print('no file {}/model/{}_{}_{}'.format(data_dir, self.role, self.model_type, self.network_type))

    # 游戏运行时把原始环境输入，分割成模型能接受的输入，在具体的模型可以修改
    def raw_env_data_to_input(self, raw_data, action):
        pass

    # 这里返回的list要和raw_env_data_to_input返回的尺寸匹配
    def empty_env(self):
        pass

    def get_record(self):
        if os.path.exists('{}/model/{}_{}_{}_record'.format(data_dir, self.role, self.model_type, self.network_type)):
            with open('{}/model/{}_{}_{}_record'.format(data_dir, self.role, self.model_type, self.network_type),
                      'r') as r:
                record = eval(r.read())
        else:
            record = {'total_epochs': 0, 'begin_time': time.asctime(time.localtime(time.time()))}
        return record

    # 分析动作的出现频率和回报率
    def operation_analysis(self, folder):
        files = os.listdir('{}/data/{}'.format(data_dir, folder))
        data_files = filter(lambda f: '.' in f, files)
        round_nums = list(set([file.split('.')[0] for file in data_files]))

        raw_env = self.raw_env_generate(folder, round_nums)
        raw_env['num'] = raw_env['num'].astype('int')
        raw_env['action'] = raw_env['action'].astype('int')
        raw_env = raw_env[raw_env['action'] > 0]
        reward_chart = raw_env.groupby(['num']).sum()['reward'].plot.line(title='reward')
        # 调整图例
        # reward_chart.legend(loc='right')
        reward_chart.legend(bbox_to_anchor=(1.0, 1.0))

        # 这里对num，action进行聚类，之后他们都在层次化索引上
        # 所以需要unstack，将action移到列名,才能绘制出按文件名分开的柱状体
        # 这里groupby(['action', 'num']) 后结果任然是有很多列，打印的时候看不出来，但操作会出错
        # count()，可以随意取一列
        action_count = raw_env.groupby(['action', 'num']).count()['reward'].unstack().fillna(0)

        # action_count 进行unstack后是Dataframe，action_total则是Series
        # 所以action_total是对第二个列聚类，使得两者列能够列对齐
        action_total = raw_env.groupby(['num']).count()['action']
        freq_chart = (action_count / action_total).plot.bar(title='freq')
        freq_chart.legend(bbox_to_anchor=(1.0, 1.0))

        # return raw_env

    # 测试模型数据是否匹配,只能训练之前用
    # 模型更新后就不准了
    def model_test(self, folder, round_nums):
        # 确定即时预测与回放的效果是一致的
        raw_env = self.raw_env_generate(folder, round_nums)
        train_env, train_index = self.train_env_generate(raw_env)
        train_reward, td_error, n_action = self.train_reward_generate(raw_env, train_env, train_index)

        # 检验后期生成的数据和当时采取的动作是否一样，注意比较的时候e_greedy 要设置成1,模型要没被训练过
        print(np.array(n_action[1]) == np.array(n_action[0]))
        # print(np.array(n_action[1]))
        # print(train_reward.max())

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
