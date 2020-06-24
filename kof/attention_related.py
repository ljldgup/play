import tensorflow.keras.backend as K
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

batch_size = 64


# SelfAttention 与 attention相比没有rnn层，直接计算权重，加权
# 如果要加残差，输出输出尺寸一样，那么参数数量应该是time_steps*seq_len*3
# Query 和 Key 在具体任务中可能是不同的，这里是self attention均采用输入、
# 这个模型应该是multihead了,输出的是和时间步同样的数量
class SelfAttention(layers.Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      # 3对应Q,K,V三个矩阵
                                      # 这里是input_shape是输入的embedding的形状,input_shape[2]嵌入向量长度
                                      # 矩阵的形状是 3,embedding_size,output_dim
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        # 一定要在最后调用它
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        # 注意这里是x最后一位，和参数的第一位相乘
        # 使用dot 所有维度都会被考虑
        # (batch_size, time_steps, embedding_size) * (embedding_size,output_dim) ->
        #                                               (batch_size,time_steps,output_dim)
        # 这里实际上是将qkv进过线性变化变成多头的部分
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        # print("WQ.shape", WQ.shape)

        # print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        # Q乘以K的转置，第一个batch size不动
        # 的keras.backend.batch_dot和tf.matmul实现功能其实是一样的智能矩阵乘法
        # 这里是time_steps,output_dim * output_dim,time_steps 最后得到time_steps,time_steps
        # 意义是每个时间步上所有时间步的得分
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        # 除以根号dk，应该是batch_size,应为张量第一维度这里是None，所以硬编码
        QK = QK / (batch_size ** 0.5)

        QK = K.softmax(QK)

        # print("QK.shape", QK.shape)

        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


# 没有找到要query的东西，self-attention用在这里又不合适,自己实现一个普通attention

class Attention(layers.Layer):

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      # 这里没有qkv，直接用一个时间步通道尺寸张量， 本质就是Timedistributed的全连接
                                      # 只有一个元素时，需要加逗号，不然返回元素，而不是元祖
                                      # 这里第一位会被消除，所以得加个1
                                      shape=(input_shape[-1], 1),
                                      initializer='uniform',
                                      trainable=True)

        # 一定要在最后调用它
        super(Attention, self).build(input_shape)

    def call(self, x):
        weight = K.dot(x, self.kernel)  # batch_size, time_steps, 1

        # 每个时间步重要程度
        # 这里softmax按二维度求和，如果不换维度，batch_size, time_steps, 1，计算出来每个time_steps的权重都是1
        weight = K.softmax(K.permute_dimensions(weight, (0, 2, 1)))
        # print("QK.shape", W.shape)
        V = K.batch_dot(weight, x)
        # 保留batch_size,后面压平
        # V = K.reshape(V.shape[0], -1)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])  # batch_size, embedding_dim


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_steps, max_dims, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        if max_dims % 2 == 1:
            max_dims += 1  # max_dims must be even
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000 ** (2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000 ** (2 * i / max_dims)).T
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))

    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_embedding[:, :shape[-2], :shape[-1]]
