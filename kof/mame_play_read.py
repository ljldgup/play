import os
import random
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout

import numpy as np

from kof.distributional_dqn import DistributionalDQN, QuantileRegressionDQN
from kof.kof_agent import RandomAgent
from kof.policy_based_model import ActorCritic, PPO, DDPG
from kof.value_based_models import DoubleDQN, DuelingDQN
from kof.kof_command_mame import operation, restart, global_set, get_action_num

'''
mame.ini keyboardprovider设置成win32不然无法接受键盘输入
autoboot_script 设置为kof.lua脚本输出对应的内存值
'''
mame_dir = 'D:/game/mame32/'


def raise_expection(thread):
    e = thread.exception()
    if e:
        raise e


def train_on_mame(model, train=True, round_num=12):
    executor = ThreadPoolExecutor(6)
    # 每次打完训练次数,太大容易极端化
    epochs = 50

    # 存放数据路径
    folder_num = 1
    while os.path.exists(str(folder_num)):
        folder_num += 1
    data_dir = os.getcwd() + '/' + str(int(folder_num))
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

    # 运行mame
    os.chdir(mame_dir)
    # 直接使用cwd参数切换目录会导致mame卡住一会，原因不明
    s = subprocess.Popen(mame_dir + '/mame.exe', bufsize=0, stdout=subprocess.PIPE, universal_newlines=True)

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
                    # 训练,第一次数据不对，是第一投币之前的数据
                    if count > 0:
                        # 保存环境，动作
                        if len(tmp_action) > 10:
                            record_file = '{}.'.format(int(count))
                            np.savetxt(data_dir + '/' + record_file + 'act', np.array(tmp_action))
                            np.savetxt(data_dir + '/' + record_file + 'env', np.array(tmp_env))
                            # 查看刚刚动作对不对
                            model.model_test(folder_num, [count])
                            if train:
                                print(str(time.asctime(time.localtime(time.time()))))
                                model.train_model(folder_num, [count], epochs=epochs)
                                if count % model.copy_interval == 0:
                                    model.save_model()
                            # 观察每次训练后的情况，如果变化过大的话，说明学习率过高
                            dqn_model.model_test(folder_num, [count])
                            print("重开")

                    tmp_action = []
                    tmp_env = []

                    # 注意这里train_interval设置成1的话那么，每次训练实际上两个模型是一样的，就是nature dqn
                    count += 1

                    if train:
                        # multi_steps 逐渐较少到1，起一定的修正效果， e_greedy增大至1
                        # 每过1/5步子减少1
                        model.multi_steps = 2 // (5 * count // round_num + 1) + 1
                        # t = count / round_num + 0.6
                        # model.e_greedy = 0.95 if t >= 1 else t
                        # 随机生成e_greedy
                        # model.e_greedy = 99.7% [-3*sigma,3*sigma] 95.4% [-2*sigma,2*sigma], 68.3% [-sigma,sigma]
                        model.e_greedy = 0.94 + 0.04 * np.random.randn()
                        # model.e_greedy = 0.5 + 0.6 * count // round_num

                    else:
                        model.e_greedy = 0.98

                    print('greedy:', model.e_greedy)

                    # 重启，role用来选人
                    restart(model.role)

                else:
                    tmp_env.append(data)

                    # 该时间步需要操作
                    if len(tmp_env) % model.operation_interval == 0:

                        # 注意tmp_action比 tmp_env长度少1， 所以这里用tmp_action判断
                        if len(tmp_action) < model.input_steps + 1:
                            keys = model.choose_action(None, None, random_choose=True)
                        else:
                            keys = model.choose_action(np.array([tmp_env[-model.input_steps:]]),
                                                       np.array([tmp_action[-model.input_steps:]]),
                                                       random_choose=False)
                        # print(keys)
                        # 按键采用一个新的线程执行，其他部分在主线程中进行，避免顺序混乱
                        # 1p 在右边
                        if data[4] > data[6]:
                            # t = executor.submit(operation, keys, True)
                            executor.submit(operation, keys, True)
                            # operation(keys, True)
                        else:
                            # t = executor.submit(operation, keys)
                            executor.submit(operation, keys)
                            # operation(keys)
                        # 提交异常，这里影响效率，暂时去掉
                        # executor.submit(raise_expection, t)
                    else:
                        # 如果不在操作的步长上，直接返回-1，不采取任何操作
                        keys = -1

                    tmp_action.append(keys)

    except:
        traceback.print_exc()
    finally:
        print(line)
        s.kill()
        model.save_model()
        executor.shutdown()
    return folder_num


if __name__ == '__main__':
    role = 'iori'

    global_set(role)
    # dqn_model = DoubleDQN('iori')
    dqn_model = PPO(role, get_action_num(role))
    # dqn_model = DuelingDQN(role, get_action_num(role))
    # QuantileRegressionDQN有bug，会过估计，暂时不明白错误在哪里
    # dqn_model = QuantileRegressionDQN()
    # dqn_model = DistributionalDQN(role, get_action_num(role))
    # dqn_model = RandomAgent('iori')
    # model.load_model('1233')
    # model = random_model('kyo')
    round_num = 42
    folder_num = train_on_mame(dqn_model, True, round_num)
    # dqn_model.train_model(folder_num, epochs=20)
    # dqn_model.save_model()

    dqn_model.operation_analysis(folder_num)
    dqn_model.model_test(folder_num, [round_num])
    dqn_model.value_test(folder_num, [1])
