import time

import win32api
import win32con

# 键位映射
keys_map_p1 = {'s': 49, 'a': 74, 'b': 75, 'c': 85, 'd': 73, '6': 68, '4': 65, '2': 83, '8': 87, 'p': 80}
keys_map_p2 = {'s': 51, 'a': 84, 'b': 89, 'c': 79, 'd': 80, '6': 78, '4': 86, '2': 66, '8': 71, 'p': 80}
keys_map = keys_map_p1
virtual_key_map = {key: win32api.MapVirtualKey(keys_map[key], 0) for key in keys_map.keys()}

direct_key_list = {1: set('24'), 2: set('2'), 3: set('26'), 4: set('4'),
                   5: set(''), 6: set('6'), 7: set('48'), 8: set('8'), 9: set('68')}
# 反方向映射
opposite_direction = {1: 3, 2: 2, 3: 1, 4: 6, 5: 5, 6: 4}

action_key_list = {0: '', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'ab', 6: 'cd', 7: 'abc'}

remain_direction = {}

# 如果实际发现很多没有写的招式，大概率是左右搞错了

# kyo
# 招数尽可能少，这样训练，实战效果反而好一些
role_commands = {}
role_commands['kyo'] = [
    # 防御
    ([4], [0]),
    # 蹲防
    ([4, 2], [0, 0]),
    # ab
    ([5], [5]),
    ([5], [6]),
    ([5], [0]),
    ([9, 0], [0, 0]),
    ([2], [1]),
    ([2, 2], [0, 4]),
    ([5], [3]),
    ([2, 3, 6], [0, 0, 1]),
    ([6, 2, 3], [0, 0, 1]),
    ([4, 2, 1], [0, 0, 2]),
    # 晴月阳，出招只有从下开始的才需要注意1,3，这边绕半圈的不需要过3
    ([6, 2, 1, 4], [0, 0, 4]),
    ([2, 3, 6, 2, 3, 6], [0, 0, 0, 0, 0, 1]),
]

# 八神出招表
role_commands['iori'] = [
    # 防御
    ([4], [0]),
    # 蹲防
    ([1], [0]),
    ([4], [2]),
    ([4], [3]),
    ([1], [1]),
    ([1], [3]),
    ([1], [4]),
    # 跳跃
    ([8, 5, 5, 5, 5], [0, 0, 0, 0, 4]),
    # ab,cd
    ([4], [5]),
    ([6], [5]),
    ([4], [6]),
    # 如果发现人物经常性不动检查是否大量输出这个动作，其次看看线程有有无异常，线程不会主动抛出错误
    ([5], [0]),
    ([6], [0]),
    ([6], [4]),
    ([6, 6], [0, 1]),
    ([2, 1, 4], [0, 0, 1]),
    ([2, 1, 4], [0, 0, 3]),
    ([2, 3, 6], [0, 0, 3]),
    ([6, 2, 1, 4], [0, 0, 0, 4]),
    ([6, 2, 3], [0, 0, 1]),
    ([2, 3, 6, 2, 1, 4], [0, 0, 0, 0, 0, 3]),
]

# 普通出招表方向加功能按键
common_commands = [
    (1, None),
    (2, None),
    (3, None),
    (4, None),
    (5, None),
    (6, None),
    (7, None),
    (8, None),
    (9, None),
    (None, 1),
    (None, 2),
    (None, 3),
    (None, 4),
    (None, 5),
    (None, 6),
    (None, 7),
]

role = ''


# 将文件中全局变量设置成chosen_role的
def global_set(chosen_role):
    global role
    role = chosen_role


def get_action_num(chosen_role):
    return len(role_commands[chosen_role])


def pause():
    win32api.keybd_event(keys_map['p'], virtual_key_map['p'], 0, 0)
    time.sleep(0.04)
    win32api.keybd_event(keys_map['p'], virtual_key_map['p'], win32con.KEYEVENTF_KEYUP, 0)


def direction_operation(direction_key, pos_reverse=False):
    global remain_direction, opposite_direction
    # 反向映射
    if pos_reverse:
        direction_key = opposite_direction[direction_key]

    # 按下新的键位，松开不用的键位，模拟摇杆, 注意这里要后送
    for key in direct_key_list[direction_key]:
        if key not in remain_direction:
            win32api.keybd_event(keys_map[key], virtual_key_map[key], 0, 0)
    time.sleep(0.01)
    for key in remain_direction:
        if key not in direct_key_list[direction_key]:
            win32api.keybd_event(keys_map[key], virtual_key_map[key], win32con.KEYEVENTF_KEYUP, 0)
    remain_direction = direct_key_list[direction_key]


def action_operation(action_key):
    # 有按键
    for key in action_key_list[action_key]:
        win32api.keybd_event(keys_map[key], virtual_key_map[key], 0, 0)
    # 不是0的话，压键长一些，0因为作为特殊技间隔，所以也需要有时间
    if action_key:
        time.sleep(0.035)
    else:
        time.sleep(0.015)

    for key in action_key_list[action_key]:
        win32api.keybd_event(keys_map[key], virtual_key_map[key], win32con.KEYEVENTF_KEYUP, 0)


def common_operation(keys):
    global remain_direction, opposite_direction
    if keys[0]:
        for key in direct_key_list[keys[0]]:
            if key not in remain_direction:
                win32api.keybd_event(keys_map[key], virtual_key_map[key], 0, 0)
        for key in remain_direction:
            if key not in direct_key_list[keys[0]]:
                win32api.keybd_event(keys_map[key], virtual_key_map[key], win32con.KEYEVENTF_KEYUP, 0)
        remain_direction = direct_key_list[keys[0]]
    elif keys[1]:
        for key in action_key_list[keys[1]]:
            win32api.keybd_event(keys_map[key], virtual_key_map[key], 0, 0)
        time.sleep(0.035)
        for key in action_key_list[keys[1]]:
            win32api.keybd_event(keys_map[key], virtual_key_map[key], win32con.KEYEVENTF_KEYUP, 0)


# 直接输入遥感和按键
def simulate(keys):
    direction_operation(keys[0] + 1)
    action_operation(keys[1])


def operation(keys, pos_reverse=False):
    # print(random_actions[num])
    global role, role_commands
    for d, a in zip(
            role_commands[role][keys][0], role_commands[role][keys][1]):
        direction_operation(d, pos_reverse)
        action_operation(a)


# 每次操作一个，选人等情况用
def operation3(keys):
    for key in keys:
        win32api.keybd_event(keys_map[key], virtual_key_map[key], 0, 0)
        time.sleep(0.1)
        win32api.keybd_event(keys_map[key], virtual_key_map[key], win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.8)
        # s.stdout.flush()


def restart():
    # 防止有键位没松开
    for key in keys_map.keys():
        time.sleep(0.2)
        win32api.keybd_event(keys_map[key], virtual_key_map[key], win32con.KEYEVENTF_KEYUP, 0)
    # 初次载入需要按任意键，不然导致后面开始按键失效
    # 选single play，advanced, 选kyo
    operation3('aass22a')
    # 有时候会卡一下，有可能是设置对手的原因
    time.sleep(2)
    operation3('aaa')
    time.sleep(2)
    operation3('aaaa')
    # 选人全部放到lua脚本进行，这里只是让程序走下去


def operation_test(role):
    global role_commands
    time.sleep(1)
    for keys in role_commands[role]:
        print(keys)
        for d, a in zip(keys[0], keys[1]):
            print(d, ' ', a)
            direction_operation(d)
            action_operation(a)
        time.sleep(1)
    direction_operation(5)
    action_operation(0)


def operation_test2(direction, action):
    time.sleep(5)
    for d, a in zip(direction, action):
        print(d, ' ', a)
        direction_operation(d)
        action_operation(a)
        # time.sleep(0.05)


def test():
    '''
      '''
    # 下蹲
    for i in range(4):
        operation_test2([2, 2, 2, 5], [0, i + 1, 0, 0])

    # 下前
    for i in range(4):
        operation_test2([2, 3, 6, 5], [0, 0, i + 1, 0])

    # 前下后
    for i in range(4):
        operation_test2([6, 3, 2, 1, 4, 5], [0, 0, 0, 0, i + 1, 0])
        time.sleep(1)

    # 前下前，升龙
    operation_test2([6, 2, 3, 5], [0, 0, 0, 1, 0])

    operation_test2([4, 2, 1, 5], [0, 0, 0, 2, 0])

    # 下前下前
    operation_test2([2, 3, 6, 2, 3, 6, 5], [0, 0, 0, 0, 0, 0, 1, 0])

    # 下下后前
    operation_test2([2, 1, 4, 2, 3, 6, 5], [0, 0, 0, 0, 0, 0, 1, 0])

    for i in range(3):
        operation_test2([5, 5, 5], [0, 0, 5 + i])
    # operation_test1([0] * 100)


def common_operation_test():
    for cmd in common_commands:
        common_operation(cmd)
        print(cmd)
        time.sleep(2)
        common_operation((5, None))


if __name__ == '__main__':
    time.sleep(2)
    # global_set('kyo')
    # test()
    operation_test('iori')
