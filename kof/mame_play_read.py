import os
import random
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from kof.distributional_dqn import DistributionalDQN
from kof.kof_agent import RandomAgent
from kof.policy_based_model import ActorCritic
from kof.value_based_models import DoubleDQN, DuelingDQN
from kof.kof_command_mame import operation, restart, simulate

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
    # 每次打完训练次数
    epochs = 60

    # 存放数据路径
    folder_num = 1

    # weight_copy间隔
    copy_interval = 6

    while os.path.exists(str(folder_num)):
        folder_num += 1
    data_dir = os.getcwd() + '/' + str(int(folder_num))
    print('数据目录：{}'.format(data_dir))
    os.mkdir(data_dir)
    with open(data_dir + '/' + model.model_name, 'w') as f:
        f.write(str(time.asctime(time.localtime(time.time()))))

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

                            # model.model_test(folder_num, [count])
                            if train:
                                model.train_model_with_sum_tree(folder_num, [count], epochs=epochs)

                    tmp_action = []
                    tmp_env = []

                    # 注意这里train_interval设置成1的话那么，每次训练实际上两个模型是一样的，就是nature dqn
                    count += 1
                    if train and count % copy_interval == 0:
                        model.save_model()
                    print("重开")

                    if train:
                        # multi_steps 逐渐较少到1，起一定的修正效果， e_greedy增大至1
                        # model.multi_steps = 4 // count + 1
                        model.e_greedy = -1 / (count + 1) + 1.1
                        # model.e_greedy = 0.6 * count / round_num + 0.6
                        print('greedy:', model.e_greedy)

                    time.sleep(4)
                    # 重启，role用来选人
                    restart(model.role)

                else:
                    tmp_env.append(data)
                    # 注意tmp_action比 tmp_env长度少1， 所以这里用tmp_action判断
                    if len(tmp_action) < model.input_steps + 1:
                        keys = model.choose_action(None, None, random_choose=True)
                    else:
                        keys = model.choose_action(np.array([tmp_env[-model.input_steps:]]),
                                                   np.array([tmp_action[-model.input_steps:]]),
                                                   random_choose=False)
                    tmp_action.append(keys)

                    # 按键采用一个新的线程执行，其他部分在主线程中进行，避免顺序混乱
                    # 1p 在右边

                    '''
                    # 使用operation_split_model时采用该函数
                    simulate(keys)
                    '''
                    if data[4] > data[6]:
                        t = executor.submit(operation, keys, True)
                        # executor.submit(operation, keys, True)
                    else:
                        t = executor.submit(operation, keys)
                        # executor.submit(operation, keys)
                    executor.submit(raise_expection, t)
                    '''
                    # 如果线程按钮不能正常工作，使用此段
                    # 捕获executor返回的异常，t为submit返回的值
                    '''

    except:
        traceback.print_exc()
    finally:
        print(line)
        s.kill()
        model.save_model()
        executor.shutdown()
    return folder_num


if __name__ == '__main__':
    # dqn_model = DoubleDQN('iori')
    # dqn_model = DuelingDQN_2('iori')
    # dqn_model = DuelingDQN('iori')
    dqn_model = RandomAgent('iori')
    # model.load_model('1233')
    # model = random_model('kyo')
    folder_num = train_on_mame(dqn_model, False)
    # dqn_model.train_model(folder_num, epochs=20)
    dqn_model.save_model()

    dqn_model.operation_analysis(folder_num)
    dqn_model.value_test(folder_num, [1])
    dqn_model.model_test(folder_num, [1])
