import os
import types

import numpy as np
import pandas as pd

from common.agent import train_model_1by1
from common.distributional_dqn_model import DistributionalDQN
from common.policy_based_model import PPO
from common.value_based_models import DuelingDQN
from kof.kof_command_mame import get_action_num
from kof.kof_network import build_rnn_attention_model, build_stacked_rnn_model

data_dir = os.getcwd()
env_colomns = ['role1_action', 'role2_action',
               'role1_energy', 'role2_energy',
               'role1_position_x', 'role1_position_y',
               'role2_position_x', 'role2_position_y', 'role1_baoqi', 'role2_baoqi', 'role1', 'role2',
               'role1_guard_value',
               'role1_combo_count',
               'role1_life', 'role2_life',
               'time', 'coins', ]
# 进入函数raw_env_data_to_input的列
input_colomns = ['role1_action', 'role2_action',
                 'role1_energy', 'role2_energy',
                 'role1_position_x', 'role1_position_y',
                 'role2_position_x', 'role2_position_y', 'role1_baoqi', 'role2_baoqi', 'action']


# 读入round_nums文件下的文件，并计算reward
def raw_env_generate(self, folder, round_nums):
    if round_nums:
        raw_env_list = []
        for i in range(0, len(round_nums)):
            if os.path.exists('{}/data/{}/{}.env'.format(data_dir, folder, round_nums[i])):
                env = np.loadtxt('{}/data/{}/{}.env'.format(data_dir, folder, round_nums[i]))
                actions = np.loadtxt('{}/data/{}/{}.act'.format(data_dir, folder, round_nums[i]))
                raw_env = pd.DataFrame(env, columns=env_colomns)
                raw_env['action'] = actions
                raw_env['num'] = round_nums[i]
                raw_env_list.append(raw_env)
            else:
                raise Exception('no {}/data/{}/{}.env'.format(data_dir, folder, round_nums[i]))

        raw_env = pd.concat(raw_env_list)
        # 重排索引，去掉重复，drop=True避免原来的索引成为列
        raw_env = raw_env.reset_index(drop=True)
        # 筛选时间在流动的列
        raw_env = raw_env[raw_env['time'].diff(1).shift(-1).fillna(1) < 0]

        # 这里改成self.operation_interval步一操作，所以diff改成对应的偏移
        role1_life_reward = raw_env['role1_life'].diff(-self.operation_interval).fillna(0)
        # 这里是向后比较所以正常返回正值，负的是由于重开造成
        role1_life_reward[role1_life_reward < 0] = 0
        role2_life_reward = raw_env['role2_life'].diff(-self.operation_interval).fillna(0)
        role2_life_reward[role2_life_reward < 0] = 0

        # 这里避免新的一局开始，血量回满被误认为报酬,在后面加输赢reward的时候已修正
        # life_reward[raw_env['time'].diff(1).fillna(1) > 0] = 0

        # combo_reward = - raw_env['role1_combo_count'].diff(-self.operation_interval).fillna(0)

        # 防守的收益
        guard_reward = -raw_env['role1_guard_value'].diff(-self.operation_interval).fillna(0)
        # 破防值回复不算
        guard_reward = guard_reward.map(lambda x: x / 10 if x > 0 else 0)

        # 值要在[-1，1]左右,reward_sum太小反而容易过估计
        # 这里自己的生命更加重要,对方生命进行缩放
        raw_env['raw_reward'] = role2_life_reward - role1_life_reward + guard_reward
        raw_env['reward'] = raw_env['raw_reward'] / self.reward_scale_factor

        # 当前步的reward实际上是上一步的，我一直没有上移，这是个巨大的错误
        # raw_env['reward'] = raw_env['reward'].shift(-1).fillna(0)
        # 这个地方应该是错误的，因为diff就是当前和之后的差，就是当前action的reward，所以应该不需要再移动

        # 根据胜负增加额外的报酬,pandas不允许切片或者搜索赋值，只能先这样
        t = raw_env[raw_env['action'] > 0]
        end_index = (t[t['time'].diff(1).shift(-1).fillna(1) > 0]).index

        for idx in end_index:
            # 这里暂时不明白为什么是loc,我是按索引取得，按理应该是iloc
            if raw_env.loc[idx, 'role1_life'] > raw_env.loc[idx, 'role2_life']:
                raw_env.loc[idx, 'reward'] = 3
            elif raw_env.loc[idx, 'role1_life'] < raw_env.loc[idx, 'role2_life']:
                raw_env.loc[idx, 'reward'] = -2

        # r值裁剪,剪裁应该放到训练数据生成的地方
        # raw_env['reward'][raw_env['reward'] > 2] = 2
        # raw_env['reward'][raw_env['reward'] < -2] = -2

        return raw_env


# 生成未加衰减奖励的训练数据
# 这里把文件夹换成env_reward_generate生成的数据，避免过度耦合
def train_env_generate(self, raw_env):
    train_env_data = self.empty_env()
    train_index = []

    for index in raw_env['reward'][raw_env['action'] != -1].index:
        # 这里是loc取的数量是闭区间和python list不一样
        # guard_value不用，放在这里先补齐
        env = raw_env[input_colomns].loc[index - self.input_steps + 1:index]

        # 去掉结尾
        if len(env) != self.input_steps or \
                raw_env['time'].loc[index - self.input_steps + 1] < raw_env['time'].loc[index]:
            pass
        else:
            data = env.values
            data = data.reshape(1, *data.shape)
            # action为了和实时输入一致，去除最后一次动作，因为这次动作是根据当前环境预测出来的
            action = env['action'].values[:-1]
            action = action.reshape(1, *action.shape)
            split_data = self.raw_env_data_to_input(data, action)
            for i in range(len(split_data)):
                train_env_data[i].append(split_data[i])
            train_index.append(index)
    train_env_data = [np.array(env) for env in train_env_data]
    # 去掉原来的长度为1的sample数量值
    train_env_data = [env.reshape(env.shape[0], *env.shape[2:]) for env in train_env_data]

    return [train_env_data, train_index]


# 游戏运行时把原始环境输入，分割成模型能接受的输入，在具体的模型可以修改
# 对应input_colomns
def raw_env_data_to_input(self, raw_data, action):
    # 将能量，爆气，上次执行的动作都只输入最后一次，作为decode的query
    return (raw_data[:, :, 0], raw_data[:, :, 1], raw_data[:, :, 2], raw_data[:, :, 3],
            raw_data[:, :, 4:6], raw_data[:, :, 6:8], raw_data[:, :, 8], raw_data[:, :, 9],

            # 注意action是上一步的，这里设置一个超长的步长，来保证只选取上一次的动作
            action[:, self.action_begin_index:self.operation_interval:])


# 这里返回的list要和raw_env_data_to_input返回的大小一样
def empty_env(self):
    return [[], [], [], [], [], [], [], [], []]


if __name__ == '__main__':
    functions = [
        build_rnn_attention_model,
        # build_stacked_rnn_model,
        raw_env_generate, train_env_generate,
        raw_env_data_to_input, empty_env]
    model1 = PPO('iori', get_action_num('iori'), functions)
    # model1 = DuelingDQN('iori', get_action_num('iori'), functions)
    # model1 = DistributionalDQN('iori', get_action_num('iori'), functions)
    # model1.model_test(4, [1])
    # t = model1.operation_analysis(3)
    # train_model_1by1(model1, [2], range(1, 4))
    # model1.save_model()
    # t = model.operation_analysis(1)

    raw_env = model1.raw_env_generate(2, [3])
    train_env, train_index = model1.train_env_generate(raw_env)
    train_reward, td_error, n_action = model1.train_reward_generate(raw_env, train_env, train_index)
    # 这里100 对应的是 raw_env 中 100+input_steps左右位置
    t = model1.predict_model.predict([np.expand_dims(env[100], 0) for env in train_env])
    # t = model.predict_model.predict([env for env in train_env])
    # output = model.output_test([ev[50].reshape(1, *ev[50].shape) for ev in train_env])
    # train_reward[range(len(n_action[1])), n_action[1]]
    # model1.model_test(1, [1])
    # t = model1.operation_analysis(5)
    # 查看训练数据是否对的上

    index = 65
    train_index[index], raw_env['action'].reindex(train_index).values[index], raw_env['reward'].reindex(
        train_index).values[index], [np.expand_dims(env[index], 0) for env in train_env]
    '''
    '''