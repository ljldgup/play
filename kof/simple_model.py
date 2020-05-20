import os

import numpy as np
from PIL import Image
import pickle

# 从指定目录下读取图片，拼接成为训练数据
from sklearn.svm import SVC, SVR


def generate_train_data(directory):
    xtrain = []
    for items in os.listdir(directory):
        im_PIL = Image.open(os.path.join(directory, items))
        xtrain.append(np.array(im_PIL))
    return xtrain


def save_simple_model(model, name):
    s = pickle.dumps(model)
    f = open(name, "wb+")
    f.write(s)
    f.close()


def read_simple_model(name):
    f2 = open(name, 'rb')
    s2 = f2.read()
    model = pickle.loads(s2)
    return model


# 用于识别continue
def if_continue_train():
    x_continue_train = generate_train_data('continue/continue')
    y_continue_train = [1] * len(x_continue_train)

    x_not_continue_train = generate_train_data('continue/not_continue')
    y_not_continue_train = [0] * len(x_not_continue_train)

    x_train = np.array(x_continue_train + x_not_continue_train)
    y_train = np.array(y_continue_train + y_not_continue_train)

    svm_x_continue_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]) / 255

    clf = SVC()
    clf.fit(svm_x_continue_train, y_train)
    print(clf.score(svm_x_continue_train, y_train))
    save_simple_model(clf, 'svm_continue_judge')
    return clf


# 这个模型每次需要加入新图片重新训练一下
def if_blood_train():
    x_blood_train = generate_train_data('blood/blood')
    y_blood_train = [1] * len(x_blood_train)

    x_no_blood_train = generate_train_data('blood/no_blood')
    y_no_blood_train = [0] * len(x_no_blood_train)

    x_train = np.array(x_blood_train + x_no_blood_train)
    y_train = np.array(y_blood_train + y_no_blood_train)

    svm_x_blood_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]) / 255

    clf = SVC()
    clf.fit(svm_x_blood_train, y_train)
    print(clf.score(svm_x_blood_train, y_train))
    save_simple_model(clf, 'svm_blood_judge')
    return clf


# 硬编码获得血量
def get_blood(pic_array):
    # 血条灰度通常在120-170之间
    is_blood = lambda x: 120 < x < 170

    # 分别放置当1/4，3/4的位置然后移动到血条边缘位置
    i = 40
    if not is_blood(pic_array[11, i]):
        while i < 160 and not is_blood(pic_array[11, i]):
            i += 1
    else:
        while i > 0 and is_blood(pic_array[11, i]):
            i -= 1
        i += 1
    p1_blood = 0
    while i < 160 and is_blood(pic_array[11, i]):
        p1_blood += 1
        i += 1

    i = 120
    if not is_blood(pic_array[11, i]):
        while i > 0 and not is_blood(pic_array[11, i]):
            i -= 1
    else:
        while i < 160 and is_blood(pic_array[11, i]):
            i += 1
        i -= 1
    p2_blood = 0
    while i > 0 and 120 < pic_array[11, i] < 170:
        p2_blood += 1
        i -= 1
    return p1_blood, p2_blood


if __name__ == '__main__':
    if_blood_train()
    #p1, p2 = get_blood(np.array(Image.open(os.path.join(r'1/0052.png'))))
