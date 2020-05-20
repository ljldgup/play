import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import tushare as ts
import time

# 解决中文显示问题
from matplotlib.ticker import MultipleLocator
from pandas.core.dtypes.common import is_string_dtype

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


#########################################################################
def data_preprocessing(*shares):
    # 涨幅
    for data in shares:
        data['pct'] = data['close'].pct_change() * 100
        # 振幅
        data['amplitude'] = abs(data['close'] - data['open']) / data['close'].shift(1) * 100


###########################################################################

# 直接用pandas对象的有时候会出bug
# sz_data['pct'].hist(bins=[-15 + 0.3 * i for i in range(120)])
# sh_data['pct'].hist(bins=[-15 + 0.3 * i for i in range(120)], density=True, cumulative=True)
def incr_density(t):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax1.hist(t['pct'], bins=[-11 + 0.1 * i for i in range(200)], density=True)
    ax1.set_title('涨幅-概率密度')

    ax2 = fig.add_subplot(212)
    ax2.grid(True)
    ax2.hist(t['pct'], bins=[-11 + 0.1 * i for i in range(200)], density=True, cumulative=True)
    ax2.set_title('涨幅-概率分布')
    plt.show()
    '''
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid(True)
    ax.hist(sh_data['pct'], bins=[-11 + 0.1 * i for i in range(200)], density=True, cumulative=True)
    '''
    return fig


###########################################################################

# 两只股票相关性散点图
def correlation_scatter(x, y):
    t = pd.merge(x, y, on='date')
    # 截取涨幅10以内, 2005年以后，深沪的涨幅
    t = t[t['pct_x'].abs() < 10][t['pct_y'].abs() < 10]
    #  t = [t['date'] > '2005-01-01']

    # 散点图
    fig = t.plot.scatter(x='pct_x', y='pct_y', s=0.5, title='相关性散点图')
    # 添加y=x直线
    x = np.linspace(-10, 10, 10)
    y = x
    fig.plot(x, y, '-r')
    plt.show()
    return fig


###########################################################################
# 多只股票（指数）相关系数
def coefficient_correlation(*shares):
    if not shares:
        return

    t = shares[0].copy()
    t.rename(columns={'pct': 'pct_0'}, inplace=True)
    for i in range(1, len(shares)):
        t = pd.merge(t[['pct_' + str(i) for i in range(i)] + ['date']], shares[i][['date', 'pct']], on='date')
        t.rename(columns={'pct': 'pct_' + str(i)}, inplace=True)
    print(t.corr())


###########################################################################
# 单只股票每日收盘价，振幅，涨幅方差
def single_volatility(data):
    # 收盘价，振幅，涨幅的
    data['close_20'] = data['close'].rolling(window=20).std()
    data['close_60'] = data['close'].rolling(window=60).std()

    data['amp_20'] = data['amplitude'].rolling(window=20).var()
    data['amp_60'] = data['amplitude'].rolling(window=60).var()

    data['pct_20'] = data['pct'].rolling(window=20).var()
    data['pct_60'] = data['pct'].rolling(window=60).var()

    if is_string_dtype(sz_data['date']):
        data['date_'] = data['date'].apply(lambda date_str: datetime.strptime(date_str, '%Y-%m-%d').date())

    items = ['close', 'amp', 'pct']
    fig = plt.figure()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    ax = [fig.add_subplot('311'), fig.add_subplot('312'), fig.add_subplot('313')]
    for i in range(3):
        # 配置横坐标
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # ax[i].xaxis.set_major_locator(mdates.AutoDateLocator)
        ax[i].xaxis.set_major_locator(MultipleLocator(len(data) / 5))
        # 自动旋转日期标记
        fig.autofmt_xdate()

        ax[i].plot(data['date_'], data[items[i] + '_60'], label=items[i] + '_60')
        ax[i].plot(data['date_'], data[items[i] + '_20'], label=items[i] + '_20')
        # 显示图例
        ax[i].legend(loc="upper right")
        ax[i].set_title(items[i] + "std")
        for tick in ax[i].get_xticklabels():
            tick.set_rotation(45)

    return fig

    # fig1 = t.plot(x='date_', y=['pct_std_20', 'pct_std_30'])
    # fig.fmt_xdata = lambda x: "{0}".format(x)
    # txt = fig.text(0.7, 0.9, '', transform=fig.transAxes)
    # fig.canvas.mpl_connect('motion_notify_event', lambda event:txt.set_text('x={0}, y={1}'.format(event.xdata, event.ydata)))


###########################################################################
# 多只股票每日收盘价，振幅，涨幅方差
def multi_volatility(*shares):
    if not shares:
        return

    length = len(shares)
    # 求所有股票日期交集，作为x轴
    common_date = shares[0]['date']

    for share in shares:
        common_date = share['date'][share['date'].isin(common_date)]

    x_date = common_date.apply(lambda date_str: datetime.strptime(date_str, '%Y-%m-%d').date())
    figs = [plt.figure(), plt.figure(), plt.figure()]
    ax = [[], figs[1].add_subplot('111'), figs[2].add_subplot('111')]

    x_major_locator = MultipleLocator(len(x_date) / 5)

    for i in range(length):
        ax[0].append(figs[0].add_subplot(str(length) + '1' + str(i)))

        shares[i]['close_std'] = shares[i]['close'].rolling(window=20).std()
        shares[i]['amp_var'] = shares[i]['amplitude'].rolling(window=20).var()
        shares[i]['pct_var'] = shares[i]['pct'].rolling(window=20).var()

        ax[0][i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax[0][i].xaxis.set_major_locator(x_major_locator)
        ax[0][i].set_title(str(i))
        ax[0][i].plot(x_date, shares[i][shares[i]['date'].isin(common_date)]['close_std'])
        ax[1].plot(x_date, shares[i][shares[i]['date'].isin(common_date)]['amp_var'], label=str(i))
        ax[2].plot(x_date, shares[i][shares[i]['date'].isin(common_date)]['pct_var'], label=str(i))

    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax[1].xaxis.set_major_locator(x_major_locator)
    ax[1].set_title("振幅方差")
    ax[1].legend(loc="upper right")

    ax[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax[2].xaxis.set_major_locator(x_major_locator)
    ax[2].set_title("涨幅方差")
    ax[2].legend(loc="upper right")

    return figs


if __name__ == '__main__':
    sz_data = pd.read_csv("sz_d_bas.csv")
    sh_data = pd.read_csv("sh_d_bas.csv")
    cyb_data = pd.read_csv("cyb_d_bas.csv")
    s_000651_data = pd.read_csv("000651_d_bas.csv")
    s_000513_data = pd.read_csv("000513_d_bas.csv")

    '''
    sz_data = pd.read_csv("sz_d_bas.csv").rename(columns = {'close': 'close_sz'})
    sh_data = pd.read_csv("sh_d_bas.csv").rename(columns = {'close': 'close_sh'})
    cyb_data = pd.read_csv("cyb_d_bas.csv").rename(columns = {'close': 'close_cyb'})
    s_000651_data = pd.read_csv("000651_d_bas.csv").rename(columns = {'close': 'close_000651'})
    s_000513_data = pd.read_csv("000513_d_bas.csv").rename(columns = {'close': 'close_000513'})
    '''
    data_preprocessing(sz_data, sh_data, cyb_data, s_000651_data, s_000513_data)

    #涨幅概率，密度分布
    #incr_density(sh_data)

    # 沪深涨幅相关性散点图
    #correlation_scatter(sz_data, sh_data)

    # 总体指数之间的相关性更强，个股之间较弱一些
    coefficient_correlation(sz_data, sh_data, cyb_data, s_000651_data)

    single_volatility(sz_data[sz_data['date'] > '2016-3'])

    # multi_volatility(sh_data[sz_data['date'] > '2016-3'], sz_data, cyb_data)

    d_002111 = ts.get_k_data('002111', ktype='d', autype='qfq', index=False,
                             start='2001-01-01', end=time.strftime("%Y-%m-%d"))

    d_000513 = ts.get_k_data('000513', ktype='d', autype='qfq', index=False,
                             start='2001-01-01', end=time.strftime("%Y-%m-%d"))

    data_preprocessing(d_002111,d_000513)

    multi_volatility(d_002111[d_002111['date'] > '2016-3'], d_000513)