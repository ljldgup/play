import pandas as pd
import numpy as np
from sklearn import svm  # svm支持向量机
from sklearn.model_selection import train_test_split

sz_data = pd.read_csv("sz_d_bas.csv")
sh_data = pd.read_csv("sh_d_bas.csv")
cyb_data = pd.read_csv("cyb_d_bas.csv")

# 分别计算5日、20日、60日的移动平均线
ma_list = [10, 20, 60]
# 计算简单算术移动平均线MA -
for ma in ma_list:
    sz_data['MA_' + str(ma)] = sz_data['close'].rolling(window=ma).mean()
    sh_data['MA_' + str(ma)] = sh_data['close'].rolling(window=ma).mean()
    cyb_data['MA_' + str(ma)] = cyb_data['close'].rolling(window=ma).mean()

sz_test = sz_data[sz_data['date'] > '2000-01-01']
sz_test['pct'] = sz_test['close'].pct_change() * 100
# 训练数据y， 涨为1 跌为0
sz_test['signal'] = sz_test['pct'].apply(lambda x: 1 if x > 0 else 0)

x_train_list = []

t = sz_test[['open', 'close', 'high', 'low', 'volume', 'MA_10', 'MA_20', 'MA_60']][sz_test['date'] > '2002-01-01']

for i in range(5, len(t)):
    x_train_list.append(t[i - 5: i].values.reshape(40))

x = np.array(x_train_list)
y = sz_test[['signal']][sz_test['date'] > '2002-01-01'][5:]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#定义感知机
clf = svm.SVC()
#使用训练数据进行训练
clf.fit(X_train, y_train)

#正确率极低
clf.score(X_test, y_test)

