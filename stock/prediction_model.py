import datetime
import random
import time

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.python.keras.layers import CuDNNGRU
from tensorflow.python.keras.optimizers import RMSprop

'''
目前影响较大的因素
1.模型天数的设置，较小模型一般较差
2.简单的模型反而效果更好。。。
3.数据中心化的初始比归一化效果好很多
4.卷积层激活函数的效果 relu < tanh < 没有激活函数，可能与权重初始化有关
5.平均池化比最大池化好
6.dense层使用relu效果较好
7.2个gru比一个gru趋势性更强，容易极端化
8.损失mse比mae更好
9.顺序打乱后效果变差
'''


# 输入日期长度,
# 输入数据量小的时候会使得预测变化的很夸张，
# 输入数据过大的时候又使新数据对后续改变不大，导致趋势不变

# 增加了卷积激活函数rule，rnn层数后长周期反而表现很差，可能是因为过拟合的原因
# 可以考虑增加一个随机生成涨跌的函数，增加可变性
# 对短周期模型，减少rnn，增加1d 卷积数量，增加对相邻k线的探测，改用max池化
# 效果很一般


# 最早才用的模型，卷积层没有激活函数，使用去均值中心化的训练效果不错
# 注意损失函数mse 比mae效果好一些
# AveragePooling1D 貌似比最大池化好一些
def build_model_1(time_steps):
    model = tf.keras.models.Sequential()
    # 时间步数太短，卷积核尺寸开始小，然后增大，不能一直用1，否则卷积无法查看相邻的关系
    model.add(layers.Conv1D(16, 2, padding='same', strides=1, input_shape=(time_steps, 5)))
    model.add(layers.Conv1D(32, 2, padding='same', strides=1))
    model.add(layers.Conv1D(64, 2, padding='same', strides=1))
    # 注意这里第二次卷积核为2的卷积实际上就已经跨过三天的k线，所以没必要用太多
    model.add(layers.Conv1D(128, 2, padding='same', strides=1))
    model.add(layers.AveragePooling1D(2))
    model.add(CuDNNGRU(128, return_sequences=True))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(5, activation='tanh'))

    # 优化算法使用adam，短周期收敛较慢
    model.compile(optimizer='adam', loss='mse')
    return model


# 尝试增加了卷积层relu激活函数，使用中心化，归一化数据，效果一般
# 比较发现即使和model_1同样的结构，使用归一化数据效果同样很差，股票数据不适合归一化
# 尝试增加了卷积层tanh激活函数，效果好一些
# 增加了一个gru层，趋势性效果放大
# 尝试加宽卷积核全连接层，效果很差
# 削减了网络规模，效果反而好了一写
def build_model_2(time_steps):
    # 卷积层过多的激活函数效果反而一般
    model = tf.keras.models.Sequential()
    # 每次输入一个月的数据量
    # 输入数据5个通道是指四个价格，加一个交易量
    # 时间步数太短，卷积核尺寸开始小，然后增大，不能一直用1，否则卷积无法查看相邻的关系
    # model.add(layers.Conv1D(16, 2, padding='same', activation='tanh', strides=1, input_shape=(time_steps, 5)))
    # model.add(layers.Conv1D(32, 2, padding='same', activation='tanh', strides=1))

    model.add(layers.Conv1D(64, 2, padding='same', activation='tanh', strides=1))
    model.add(layers.Conv1D(128, 2, padding='same', activation='tanh', strides=1))
    model.add(layers.AveragePooling1D(2))
    # 卷积核数量作为需要标准化的轴，每个卷积核使用不用beta和gamma
    # 在这里用BN层效果一般，可能是股价的均值，方差不稳定
    # model.add(layers.BatchNormalization(axis=2))
    # activation = 'relu' CuDNNGRU，CuDNNLSTM的激活函数貌似是内定的
    # 单次直接输入多个日期，貌似不需要时间记忆，先去掉,把神经网络加深。。。。
    # return_sequences 决定返回单个 hidden state值还是返回全部time steps 的 hidden state值
    # 这里第一个不加return sequence ，不能连7续用两个gru，输出没有time steps形状不匹配
    model.add(CuDNNGRU(128, return_sequences=True))
    # model.add(CuDNNGRU(256, return_sequences=True))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(5, activation='tanh'))
    # 最终输出层用tanh收敛好于relu，并且预测效果远好于relu，可能是输出-1，1，对应输入范围广的原因
    model.compile(optimizer='adam', loss='mse')
    return model


# 使用卷积层relu激活,配合uniform权重初始化，结果尚可，可以考虑加深网络
def build_model_3(time_steps):
    model = tf.keras.models.Sequential()

    model.add(layers.Conv1D(64, 2, padding='same', strides=1, activation='relu', kernel_initializer='uniform',
                            input_shape=(time_steps, 5)))
    # model.add(layers.Conv1D(32, 2, padding='same', strides=1,activation='relu',kernel_initializer='uniform'))
    # model.add(layers.Conv1D(64, 2, padding='same', strides=1,activation='relu',kernel_initializer='uniform'))
    model.add(layers.Conv1D(128, 2, padding='same', strides=1, activation='relu', kernel_initializer='uniform'))
    model.add(layers.AveragePooling1D(2))
    model.add(CuDNNGRU(128, return_sequences=True))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(5, activation='tanh'))

    # 优化算法使用adam，短周期收敛较慢
    model.compile(optimizer='adam', loss='mse')
    return model


# 一次生成batch_size个数据
def data_generator(data, time_steps, batch_size=20):
    # 之所以要多减1，是留一次预测，x直接取到最后，就没有预测值了
    seq = list(range(0, len(data) - time_steps - batch_size - 1, batch_size))

    # 随机打乱一下获取顺序
    # 考虑lstm的情况下先不打乱
    # lstm 的time steps 在单个数据内，通常是（samples，step，....   所以可以打乱, 但打乱后不知道为什么效果很差
    # random.shuffle(seq)

    for st in seq:
        t_x_data = []
        t_y_data = []
        for i in range(st, st + batch_size):
            train_x = data[i:i + time_steps].copy()
            train_y = data[i + time_steps:i + time_steps + 1].copy()

            # 归一化处理
            v_max = train_x.max()
            v_min = train_x.min()
            v_mean = train_x.mean()
            # 波动百分比统一处理
            """
            v_max[1:-1] = v_max[1:-1].max()
            v_min[1:-1] = v_min[1:-1].min()
            v_mean[1:-1] = v_mean[1:-1].mean()
            """
            # pandas能一次处理多列
            train_x = (train_x - v_mean) / (v_max - v_min)
            train_y = (train_y - v_mean) / (v_max - v_min)

            # 考虑到rule会抛弃值<0的情况，所以改成缩放到0-1之间,注意get_future 那里也要改
            # train_x = (train_x - v_min) / (v_max - v_min)
            # train_y = (train_y - v_min) / (v_max - v_min)
            t_x_data.append(train_x.values)
            t_y_data.append(train_y.values)

        x = np.array(t_x_data)
        # y 多了一个维度
        y = np.array(t_y_data).reshape(batch_size, 5)
        yield x, y


# 模型作为参数用于继续训练
# epoch过大产生过拟合，反而逼真程度一般
def train_model(predict_model, data, time_steps, epoch_num=10):
    print('train on {} steps'.format(time_steps))
    loss_history = []
    for epoch in range(epoch_num):
        time.sleep(1)
        print("epoch ", epoch)
        loss = 0
        # 尝试逐步增加batch的规模
        for x, y in data_generator(data, time_steps, batch_size=30 + epoch):
            loss += predict_model.train_on_batch(x, y)
            # print(batch_num, ' loss:', loss)

        print('loss:', loss)
        loss_history.append(loss)
    return loss_history


def get_future_one_day(data, predict_model, time_steps):
    # 不覆盖原数据
    x_data = data[['open', 'open_close_pct', 'open_high_pct', 'open_low_pct', 'volume']][-time_steps:].copy()
    # 归一化处理
    v_max = x_data.max()
    v_min = x_data.min()
    v_mean = x_data.mean()

    # 百分比同一处理
    """
    v_max[1:-1] = v_max[1:-1].max()
    v_min[1:-1] = v_min[1:-1].min()
    v_mean[1:-1] = v_mean[1:-1].mean()
    """
    x_data = (x_data - v_mean) / (v_max - v_min)
    # x_data = (x_data - v_min) / (v_max - v_min)
    # 预测并回复数据尺度
    next_day = predict_model.predict(x_data.values.reshape(1, time_steps, 5))
    # 追加到最后,日期栏经过后面操作后，会变成NaN，是什么值无所谓

    # 恢复到正常尺度
    next_day = next_day * (v_max.values - v_min.values) + v_mean.values
    # next_day = next_day * (v_max.values - v_min.values) + v_min.values
    return next_day


# 获取未来数天的预测
#
def get_future(data, predict_models, time_steps_list, days=10):
    data = data.copy()
    data_length = len(data)
    if data_length < time_steps_list[0]:
        return

    last_date = datetime.datetime.strptime(data.iloc[- 1]['date'], "%Y-%m-%d")

    for i in range(days):
        next_days = []
        for model, step in zip(predict_models, time_steps_list):
            next_days.append(get_future_one_day(data, model, step))
        # 随机策略。。
        # next_day_average = next_days[random.randint(0, len(next_days) - 1)]
        next_day_average = sum(next_days) / len(next_days)
        last_date += datetime.timedelta(days=1)
        next_day = [last_date.strftime("%Y-%m-%d")]

        # 一般取均值效果更加自然一些
        # 长期模型，短期模型的均值
        next_day.extend(next_day_average.tolist()[0])

        '''
        # 长期模型提供价格成交量，短期模型提供日内波动百分比
        next_day.append(next_day_long_term[0][0])
        next_day.extend(next_day_short_term.tolist()[0][1:-1])
        next_day.append(next_day_long_term[0][-1])
        '''
        data.loc[data_length] = next_day
        data_length += 1

    return data


if __name__ == '__main__':
    # 时间步过短拟合的效果一般，容易出现极端走势
    # 目前来看 >=60的 效果不错， <=40 效果不佳
    time_steps = 60
    model_1 = build_model_1(time_steps)
    model_2 = build_model_2(time_steps)
    model_3 = build_model_2(time_steps)
    # cyb_data = ts.get_k_data('cyb', ktype='D', autype='qfq', index=False,
    #                           start='2001-01-01', end=time.strftime("%Y-%m-%d"))
    # cyb_data.to_csv("cyb_d_bas.csv")

    index_data = pd.read_csv("cyb_d_bas.csv")
    index_data = index_data[['date', 'open', 'close', 'high', 'low', 'volume']]
    index_data['open_close_pct'] = (index_data['open'] - index_data['close']) / index_data['open']
    index_data['open_high_pct'] = (index_data['open'] - index_data['high']) / index_data['open']
    index_data['open_low_pct'] = (index_data['open'] - index_data['low']) / index_data['open']
    index_data['open_low_pct'] = (index_data['open'] - index_data['low']) / index_data['open']
    # 尝试使用成交量幅度代替成交量，效果很差
    # index_data['volume_pct'] = index_data['volume'].pct_change()

    index_data = index_data[['date', 'open', 'open_close_pct', 'open_high_pct', 'open_low_pct', 'volume']]

    train_data = index_data[['open', 'open_close_pct', 'open_high_pct', 'open_low_pct', 'volume']]

    # 训练或者读取权重

    loss_1 = train_model(model_1, train_data, time_steps)
    model_1.save_weights('model_1_prediction_{}'.format(time_steps))
    # model_1.load_weights('model_1_prediction_{}'.format(time_steps))

    loss_2 = train_model(model_2, train_data, time_steps)
    model_2.save_weights('model_2_prediction_{}'.format(time_steps))
    # model_2.load_weights('model_2_prediction_{}'.format(time_steps))

    loss_3 = train_model(model_3, train_data, time_steps)
    model_3.save_weights('model_3_prediction_{}'.format(time_steps))
    # model_3.load_weights('model_3_prediction_{}'.format(time_steps))

    '''
    # long_term_model.load_weights('long_term_model_prediction_{}'.format(long_term_steps))
    # short_term_model.load_weights('short_term_model_prediction_{}'.format(short_term_steps))
    '''

    future_days = 100

    # 尝试使用长短周期来混合使用，分别预测价格成交量， 和日内波动，但效果一般，
    # 主要是短周期的模型拟合较差，可能两者不能采用同一模型
    # 想看单独一个模型预测效果直接把列表里换成一个就可以了
    cyb_future_data = get_future(index_data, [model_3, model_2, model_1],
                                 [time_steps, time_steps, time_steps],
                                 future_days)

    cyb_future_data['close'] = cyb_future_data['open'] - cyb_future_data['open'] * cyb_future_data['open_close_pct']
    cyb_future_data['high'] = cyb_future_data['open'] - cyb_future_data['open'] * cyb_future_data['open_high_pct']
    cyb_future_data['low'] = cyb_future_data['open'] - cyb_future_data['open'] * cyb_future_data['open_low_pct']

    # 直接采用一些echart的实验网页查看效果
    # 这里dataframe-array-list,写入文件便于直接在复制粘贴到网页中展示
    with open('predict_data', 'w') as f:
        f.write(str(cyb_future_data[['date', 'open', 'close', 'high', 'low', 'volume']][
                    -future_days - future_days // 2:].values.tolist()).replace(', [', ',\n['))
