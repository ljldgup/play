import os
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common.agent import RandomAgent
from common.distributional_dqn_model import DistributionalDQN
from common.policy_based_model import PPO
from common.value_based_models import DuelingDQN
from kof.kof_agent import raw_env_generate, train_env_generate, empty_env, raw_env_data_to_input
from kof.kof_command_mame import global_set, get_action_num, stdin_commands, opposite_direction
from kof.kof_network import build_rnn_attention_model, build_stacked_rnn_model, build_multi_attention_model

'''
mame.ini keyboardprovider设置成win32不然无法接受键盘输入
autoboot_script 设置为kof.lua脚本输出对应的内存值
'''
mame_dir = r'D:\Program Files (x86)\mame'
# 临时存放数据
tmp_action = []
tmp_env = []
commands = []
count = 0
epochs = 40
train_interval = 3
folder_num = 1
data_dir = os.getcwd() + '/data/'


def raise_expection(thread):
    e = thread.exception()
    if e:
        raise e


def train_on_mame(model, train=True, round_num=15):
    global data_dir, tmp_env, tmp_action, folder_num
    init(model)
    # 这里不指定stdin,stdout没法用
    mame = subprocess.Popen(mame_dir + '/mame.exe', bufsize=0, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, universal_newlines=True)

    try:
        while count <= round_num:
            line = mame.stdout.readline()
            # [:-1]去掉\n
            data = list(map(float, line[:-1].split("\t")))

            # 根据4个币判断是否game over，输了直接重新开始不续币
            if data[-1] == 4:
                process_after_each_round(model, train)

                # 重启,mame得到输入后自动重新选人，在lua中实现
                mame.stdin.write("0\n")
                mame.stdin.flush()
            else:
                running(mame, model, data)
    except:
        traceback.print_exc()
    finally:
        mame.terminate()
        model.save_model()
        # executor.shutdown()
    return folder_num, count - 1


def init(model):
    global data_dir, folder_num, count

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # 存放数据路径
    folder_num = 1
    count = 0
    while os.path.exists(data_dir + str(folder_num)):
        folder_num += 1
    data_dir = data_dir + str(int(folder_num))

    print('数据目录：{}'.format(data_dir))
    os.mkdir(data_dir)

    save_model_info(model)

    # 运行mame,直接使用cwd参数切换目录会导致mame卡住一会，原因不明
    os.chdir(mame_dir)


def save_model_info(model):
    # 不是随机模型保存模型属性
    if type(model) is not RandomAgent:
        with open(data_dir + '/' + model.model_type, 'w') as f:
            f.writelines([str(time.asctime(time.localtime(time.time()))) + '\n',
                          model.model_type + '\n',
                          model.network_type + '\n',
                          'reward_scale_factor {}\n'.format(model.reward_scale_factor),
                          'copy interval {}\n'.format(model.copy_interval),
                          'input steps {}\n'.format(model.input_steps),
                          'action num {}\n'.format(model.action_num),
                          'reward decay {}\n'.format(model.reward_decay),
                          'learning rate {}\n'.format(model.lr),
                          'multisteps {}\n'.format(model.multi_steps)]
                         )
            f.write(str(model.record))
            f.write('\n---------\n')
            # write不会换行
            model.predict_model.summary(print_fn=lambda x: f.write('{}\n'.format(x)))


def process_after_each_round(model, train):
    global data_dir, tmp_env, tmp_action, count
    # 这里训练过长的时间，mame会卡死，目前不知道什么原因
    # 后续发现是重启过程中mame输出过多导致程序卡死，已经在lua脚本中进行修正，只输出一次
    # 发现重启前时间过长，某些模拟打斗，动画也会造成卡顿
    if count > 0:
        # 保存环境，动作
        if len(tmp_action) > 10:
            np.savetxt('{}/{}.act'.format(data_dir, count), np.array(tmp_action))
            np.savetxt('{}/{}.env'.format(data_dir, count), np.array(tmp_env))
            # 查看刚刚动作对不对，因为e-greedy会有少数随机，如果很大比例是false说明训练数据和实时数据不一样
            # PPO查看时要把按概率采样的函数注释掉，用原来的argmax版本
            print('实时动作与训练数据输出动作比较 {}'.format(count))
            # model.model_test(folder_num, [count])

            if train and count % 1 == 0:
                print(str(time.asctime(time.localtime(time.time()))))
                model.train_model(folder_num, [count], epochs=epochs)
                # 注意copy_interval设置成1的话那么，每次训练实际上两个模型是一样的，就是nature dqn

            if count % model.copy_interval == 0:
                model.save_model()
            # 观察每次训练后的情况，如果变化过大的话，说明学习率太大
            # print('训练后动作变化情况')
            # dqn_model.model_test(folder_num, [count])
    print("重开")
    tmp_action = []
    tmp_env = []
    count += 1

    if train:
        # 每过1/5步子减少1
        # multi_steps = 2 // (5 * count // round_num + 1) + 1
        multi_steps = 2
        if model.model_type.startswith('PPO'):
            model.critic.multi_steps = multi_steps
        else:
            model.multi_steps = multi_steps
        # 随机生成e_greedy
        # model.e_greedy = 99.7% [-3*sigma,3*sigma] 95.4% [-2*sigma,2*sigma], 88.3% [-sigma,sigma]
        model.e_greedy = 0.91 + 0.03 * np.random.randn()
        # model.e_greedy = 0.5 + 0.8 * count // round_num

    else:
        model.e_greedy = 1

    print('greedy:', model.e_greedy)


def running(mame, model, data):
    global data_dir, tmp_env, tmp_action, count, commands

    tmp_env.append(data)
    choosed = False

    if not commands:
        # 注意tmp_action比 tmp_env长度少1， 所以这里用tmp_action判断
        if len(tmp_action) < model.input_steps:
            cmd_index = model.choose_action(None, None, random_choose=True)
        else:
            cmd_index = model.choose_action(np.array([tmp_env[-model.input_steps:]]),
                                            np.array([tmp_action[- model.input_steps:]]),
                                            random_choose=False)
        commands = list(stdin_commands[cmd_index])

        choosed = True
        tmp_action.append(cmd_index)
        print()

    if commands:
        cmd = commands.pop(0)
    else:
        # 第0个commands为空, 执行方向5，
        cmd = 5

    # 用于调试
    # key = input()
    # if not key:
    #     key = 5

    if data[2] > data[4]:
        cmd = opposite_direction[cmd]
    print(cmd, end=",")

    mame.stdin.write(str(cmd) + '\n')
    mame.stdin.flush()

    if not choosed:
        tmp_action.append(-1)


if __name__ == '__main__':
    role = 'iori'

    global_set(role)
    # dqn_model = DoubleDQN('iori')
    functions = [build_stacked_rnn_model,
                 # build_rnn_attention_model,
                 # build_multi_attention_model,
                 raw_env_generate, train_env_generate,
                 raw_env_data_to_input, empty_env]
    dqn_model = PPO(role='iori', action_num=get_action_num(), functions=functions)
    # dqn_model = DuelingDQN('iori', get_action_num(), functions)
    # dqn_model = DistributionalDQN('iori', get_action_num('iori'), functions)
    # QuantileRegressionDQN有bug，会过估计，暂时不明白错误在哪里
    # dqn_model = QuantileRegressionDQN()
    # dqn_model = RandomAgent(role='iori', action_num=get_action_num())
    # # 公司电脑第一次预测特别慢，所以先预测一次在训练
    # dqn_model.predict_model.predict(
    #     [np.random.randn(1, 8), np.random.randn(1, 8), np.random.randn(1, 8), np.random.randn(1, 8),
    #      np.random.randn(1, 8, 4), np.random.randn(1, 8), np.random.randn(1, 8), np.random.randn(1, 8)])
    round_num = 40
    # folder_num, count = train_on_mame(dqn_model, False)
    folder_num, count = train_on_mame(dqn_model, True, round_num)
    # dqn_model.train_model(folder_num, epochs=20)
    # dqn_model.save_model()

    dqn_model.operation_analysis(folder_num)
    dqn_model.model_test(folder_num, [count])
    dqn_model.value_test(folder_num, [count])
