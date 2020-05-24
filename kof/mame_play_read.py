import os
import random
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from kof.dqn_models import model_1, dueling_dqn_model, random_model, model_2
from kof.kof_command_mame import operation, restart, simulate
from kof.operation_split_model import operation_split_model

'''
mame.ini keyboardprovider设置成win32不然无法接受键盘输入
autoboot_script 设置为kof.lua脚本输出对应的内存值
'''
mame_dir = 'D:/game/mame32/'


def train_on_mame(model, train=True, round_num=12):
    executor = ThreadPoolExecutor(3)
    # 每次打完训练次数
    epochs = 60

    # 存放数据路径
    folder_num = 1

    # 训练间隔
    train_interval = 1

    while os.path.exists(str(folder_num)):
        folder_num += 1
    data_dir = os.getcwd() + '/' + str(int(folder_num))
    print('数据目录：{}'.format(data_dir))
    os.mkdir(data_dir)

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
                # print(data)

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
                                if count % train_interval == 0:
                                    model.train_model(folder_num, range(count - train_interval + 1, count + 1),
                                                      epochs=epochs)
                                    model.save_model()
                                model.e_greedy = count / round_num * 0.3 + random.random() * 0.2 + 0.6
                            else:
                                model.e_greedy = 0.99

                    tmp_action = []
                    tmp_env = []
                    count += 1

                    print("重开")
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
                        # t = executor.submit(operation, keys, True)
                        executor.submit(operation, keys, True)
                    else:
                        # t = executor.submit(operation, keys)
                        executor.submit(operation, keys)
                    '''
                    # 如果线程按钮不能正常工作，使用此段
                    # 捕获executor返回的异常，t为submit返回的值
                    e = t.exception()
                    if e:
                        raise e
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
    # model = operation_split_model('iori')
    dqn_model = dueling_dqn_model('iori')
    # dqn_model = model_2('iori')
    # dqn_model = model_1('iori')
    # model.load_model('1233')
    # model = random_model('kyo')
    folder_num = train_on_mame(dqn_model, False)
    dqn_model.train_model(folder_num, epochs=60)
    dqn_model.save_model()

    dqn_model.operation_analysis(folder_num)
