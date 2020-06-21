import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import concatenate, BatchNormalization, CuDNNLSTM

batch_size = 64


def build_attention_model(input_steps, action_num):
    role1_actions = Input(shape=(input_steps,), name='role1_actions')
    role2_actions = Input(shape=(input_steps,), name='role2_actions')
    role1_actions_embedding = layers.Embedding(512, 8, name='role1_actions_embedding')(role1_actions)
    role2_actions_embedding = layers.Embedding(512, 8, name='role2_actions_embedding')(role2_actions)

    role1_energy = Input(shape=(1,), name='role1_energy')
    role1_energy_embedding = layers.Embedding(5, 2, name='role1_energy_embedding')(role1_energy)
    role2_energy = Input(shape=(1,), name='role2_energy')
    role2_energy_embedding = layers.Embedding(5, 2, name='role2_energy_embedding')(role2_energy)

    role1_baoqi = Input(shape=(1,), name='role1_baoqi')
    role1_baoqi_embedding = layers.Embedding(2, 1, name='role1_baoqi_embedding')(role1_baoqi)
    role2_baoqi = Input(shape=(1,), name='role2_baoqi')
    role2_baoqi_embedding = layers.Embedding(2, 1, name='role2_baoqi_embedding')(role2_baoqi)

    role1_x_y = Input(shape=(input_steps, 2), name='role1_x_y')
    role2_x_y = Input(shape=(input_steps, 2), name='role2_x_y')

    role_position = concatenate([role1_x_y, role2_x_y])
    role_position = BatchNormalization()(role_position)

    role_distance = layers.Subtract()([role1_x_y, role2_x_y])
    role_distance = BatchNormalization()(role_distance)

    # 使用attention模型
    time_related_layers = [role1_actions_embedding, role_position, role_distance,
                           role2_actions_embedding]
    attention_output = []

    # 理论上在self attention外面包一层 rnn就是attention，这边暂时这么干
    # self attention 是自注意力，query也是自己，语言翻译中用于注意上下文语境，但这里很明显有问题
    for layer in time_related_layers:
        t = CuDNNLSTM(256, return_sequences=True)(layer)
        t = Attention()(t)
        # Attention输出了单个时间步,也可以用squeeze
        t = layers.Flatten()(t)
        attention_output.append(t)
    t_status = layers.concatenate(attention_output)

    t_status = layers.concatenate(
        [t_status, K.squeeze(role1_baoqi_embedding, 1), K.squeeze(role2_baoqi_embedding, 1),
         K.squeeze(role1_energy_embedding, 1),
         K.squeeze(role2_energy_embedding, 1)])
    t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
    t_status = BatchNormalization()(t_status)
    output = layers.LeakyReLU(0.05)(t_status)

    # 曝气应该是个综合性影响，所以直接加在最后

    shared_model = Model([role1_actions, role2_actions, role1_energy, role2_energy,
                          role1_x_y, role2_x_y, role1_baoqi, role2_baoqi], output)
    # 这里模型不能编译，不然后面无法扩充
    return shared_model


# SelfAttention 与 attention相比没有rnn层，直接计算权重，加权
# 如果要加残差，输出输出尺寸一样，那么参数数量应该是time_steps*seq_len*3
# Query 和 Key 在具体任务中可能是不同的，这里是self attention均采用输入
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
                                      # 矩阵的形状是 3,time_steps,output_dim
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        # 一定要在最后调用它
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        # 注意这里是x的第三位开始和kernel相乘，第1位是batch_size, 第二位是time_steps
        # 使用dot 所有维度都会被考虑
        # (None, 4, 5) * (5, 6) -> (1, 4, 6)
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape", WQ.shape)

        # print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        # Q乘以K的转置，第一个batch size不动
        # 的keras.backend.batch_dot和tf.matmul实现功能其实是一样的智能矩阵乘法
        # 这里是4,5 x 5,4 最后得到4x4,意义是每个时间步上所有时间步的得分
        # 这里决定输出的是
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        # 除以根号dk，应该是batch_size,应为张量第一维度这里是None，所以硬编码
        QK = QK / (batch_size ** 0.5)

        QK = K.softmax(QK)

        print("QK.shape", QK.shape)

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


if __name__ == '__main__':
    '''

    input_steps = 4
    input = layers.Input(shape=(input_steps,))
    input_embedding = layers.Embedding(10, 5)(input)
    output = SelfAttention(6)(input_embedding)
    model = Model(input, output)

    # 不编译也可以直接运行，不过速度慢
    model.predict(np.array([[1, 2, 3, 4]]))
    '''
    t = build_attention_model(8, 20)
