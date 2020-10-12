import os
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common.distributional_dqn_model import DistributionalDQN
from common.policy_based_model import PPO
from common.value_based_models import DuelingDQN
from kof.kof_agent import raw_env_generate, train_env_generate, empty_env, raw_env_data_to_input
from kof.kof_command_mame import operation, restart, global_set, get_action_num, common_operation, common_commands, \
    pause
from kof.kof_network import build_rnn_attention_model, build_stacked_rnn_model, build_multi_attention_model

'''
mame.ini keyboardprovider设置成win32不然无法接受键盘输入
autoboot_script 设置为kof.lua脚本输出对应的内存值
'''
mame_dir = 'D:/game/mame/'


def raise_expection(thread):
    e = thread.exception()
    if e:
        raise e


def train_on_mame(model, train=True, round_num=15):
    executor = ThreadPoolExecutor(6)
    # 每次打完训练次数,太大容易极端化
    epochs = 40
    train_interval = 3
    data_dir = os.getcwd() + '/data/'
    # 存放数据路径
    folder_num = 1
    while os.path.exists(data_dir + str(folder_num)):
        folder_num += 1
    data_dir = data_dir + str(int(folder_num))

    print('数据目录：{}'.format(data_dir))
    os.mkdir(data_dir)
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

    # 临时存放数据
    tmp_action = []
    tmp_env = []
    # 运行mame,直接使用cwd参数切换目录会导致mame卡住一会，原因不明
    os.chdir(mame_dir)
    s = subprocess.Popen(mame_dir + '/mame64.exe', bufsize=0, stdout=subprocess.PIPE, universal_newlines=True)
    count = 0
    try:
        while count <= round_num:
            line = s.stdout.readline()
            if line:
                # 去掉\n
                data = list(map(float, line[:-1].split(" ")))

                # 根据4个币判断是否game over，输了直接重新开始不续币
                if data[-1] == 4:

                    # 这里训练过长的时间，mame会卡死，目前不知道什么原因
                    # 后续发现是重启过程中mame输出过多导致程序卡死，已经在lua脚本中进行修正，只输出一次
                    # 发现重启前时间过长，某些模拟打斗，动画也会造成卡顿
                    if count > 0:
                        # 暂停游戏，游戏在运行一定时间后，内存容易出现不可控情况，导致卡顿
                        pause()
                        # 保存环境，动作
                        if len(tmp_action) > 10:
                            np.savetxt('{}/{}.act'.format(data_dir, count), np.array(tmp_action))
                            np.savetxt('{}/{}.env'.format(data_dir, count), np.array(tmp_env))
                            # 查看刚刚动作对不对，因为e-greedy会有少数随机，如果很大比例是false说明训练数据和实时数据不一样
                            # PPO查看时要把按概率采样的函数注释掉，用原来的argmax版本
                            print('实时动作与训练数据输出动作比较 {}'.format(count))
                            model.model_test(folder_num, [count])

                            if train and count > 1 and (count - 1) % train_interval == 0:
                                print(str(time.asctime(time.localtime(time.time()))))
                                model.train_model(folder_num, list(range(count - train_interval, count)),
                                                  epochs=epochs)
                                # 注意copy_interval设置成1的话那么，每次训练实际上两个模型是一样的，就是nature dqn

                            if count % model.copy_interval == 0:
                                model.save_model()
                            # 观察每次训练后的情况，如果变化过大的话，说明学习率太大
                            print('训练后动作变化情况')
                            dqn_model.model_test(folder_num, [count])
                            pause()  # 开始游戏
                    print("重开")
                    tmp_action = []
                    tmp_env = []
                    count += 1

                    if train:
                        # 每过1/5步子减少1
                        # multi_steps = 2 // (5 * count // round_num + 1) + 1
                        multi_steps = 4
                        if model.model_type.startswith('PPO'):
                            model.critic.multi_steps = multi_steps
                        else:
                            model.multi_steps = multi_steps
                        # 随机生成e_greedy
                        # model.e_greedy = 99.7% [-3*sigma,3*sigma] 95.4% [-2*sigma,2*sigma], 68.3% [-sigma,sigma]
                        model.e_greedy = 0.91 + 0.03 * np.random.randn()
                        # model.e_greedy = 0.5 + 0.6 * count // round_num

                    else:
                        model.e_greedy = 1

                    print('greedy:', model.e_greedy)

                    # 重启，role用来选人
                    restart()

                else:
                    tmp_env.append(data)

                    # 该时间步需要操作
                    if len(tmp_env) % model.operation_interval == 0:

                        # 注意tmp_action比 tmp_env长度少1， 所以这里用tmp_action判断
                        if len(tmp_action) < model.input_steps:
                            # 开局蹲防
                            keys = model.choose_action(None, None, random_choose=True)
                        else:
                            keys = model.choose_action(np.array([tmp_env[-model.input_steps:]]),
                                                       np.array([tmp_action[-model.input_steps:]]),
                                                       random_choose=False)

                        # common_commands通用按键，无需分左右，交给网络自己判断
                        # executor.submit(common_operation, common_commands[keys])
                        # print(keys, ':', line)

                        # 按键采用一个新的线程执行，
                        if data[4] > data[6]:  # 1p 在右边
                            # t = executor.submit(operation, keys, True)
                            executor.submit(operation, keys, True)
                        else:
                            # t = executor.submit(operation, keys)
                            executor.submit(operation, keys)
                        # 没动做很可能是线程异常，这里不会报，需要提交异常，这里影响效率，有必要再用
                        # executor.submit(raise_expection, t)
                    else:
                        # 如果不在操作的步长上，直接返回-1，不采取任何操作
                        # -1代表没有任何操作，而0,5等都是回中或者防御，回对当前状态造成影响
                        keys = -1
                    tmp_action.append(keys)
    except:
        traceback.print_exc()
    finally:
        print(line)
        s.terminate()
        model.save_model()
        executor.shutdown()
    return folder_num, count - 1


if __name__ == '__main__':
    role = 'iori'

    global_set(role)
    # dqn_model = DoubleDQN('iori')
    functions = [build_stacked_rnn_model,
                 # build_rnn_attention_model,
                 # build_multi_attention_model,
                 raw_env_generate, train_env_generate,
                 raw_env_data_to_input, empty_env]
    dqn_model = PPO('ioriVSkyo', get_action_num('iori'), functions)
    # dqn_model = DuelingDQN('iori', get_action_num('iori'), functions)
    # dqn_model = DistributionalDQN('iori', get_action_num('iori'), functions)
    # QuantileRegressionDQN有bug，会过估计，暂时不明白错误在哪里
    # dqn_model = QuantileRegressionDQN()
    # dqn_model = RandomAgent('iori')
    round_num = 40
    # folder_num, count = train_on_mame(dqn_model, False)
    folder_num, count = train_on_mame(dqn_model, True, round_num)
    # dqn_model.train_model(folder_num, epochs=20)
    # dqn_model.save_model()

    dqn_model.operation_analysis(folder_num)
    dqn_model.model_test(folder_num, [count])
    dqn_model.value_test(folder_num, [count])
