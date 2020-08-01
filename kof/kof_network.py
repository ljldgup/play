from tensorflow.python.keras import layers
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import concatenate, BatchNormalization, CuDNNLSTM
from tensorflow.keras import backend as K
from common.attention_related import Attention, SelfAttention, BahdanauAttention


# 输入配合raw_env_data_to_input使用
def general_input(self):
    role1_actions = Input(shape=(self.input_steps,), name='role1_actions')
    role2_actions = Input(shape=(self.input_steps,), name='role2_actions')
    # 鉴于embedding就是onehot+全连接，这里加大embedding的size
    role1_actions_embedding = layers.Embedding(512, 16, name='role1_actions_embedding')(role1_actions)
    role2_actions_embedding = layers.Embedding(512, 16, name='role2_actions_embedding')(role2_actions)

    role1_energy = Input(shape=(1,), name='role1_energy')
    role1_energy_embedding = layers.Embedding(5, 2, name='role1_energy_embedding')(role1_energy)
    role2_energy = Input(shape=(1,), name='role2_energy')
    role2_energy_embedding = layers.Embedding(5, 2, name='role2_energy_embedding')(role2_energy)

    role1_baoqi = Input(shape=(1,), name='role1_baoqi')
    role1_baoqi_embedding = layers.Embedding(2, 1, name='role1_baoqi_embedding')(role1_baoqi)
    role2_baoqi = Input(shape=(1,), name='role2_baoqi')
    role2_baoqi_embedding = layers.Embedding(2, 1, name='role2_baoqi_embedding')(role2_baoqi)

    role1_x_y = Input(shape=(self.input_steps, 2), name='role1_x_y')
    role2_x_y = Input(shape=(self.input_steps, 2), name='role2_x_y')

    # 感觉这种环境每次都不同，小批量数据bn可能不太稳定，这里先不用，直接归一化
    role_position = concatenate([role1_x_y, role2_x_y])
    # normal_role_position = BatchNormalization()(role_position)
    # 这里加dense就是对最后一层坐标进行全连接，和timedistribute相同
    conv_role_position = layers.Conv1D(filters=8, kernel_size=2, strides=1, padding='same')(role_position)
    # conv_role_position = BatchNormalization()(conv_role_position)
    role_distance = layers.Subtract()([role1_x_y, role2_x_y])
    # normal_role_distance = BatchNormalization()(role_distance)
    role_abs_distance = layers.Lambda(lambda x: K.abs(x))(role_distance)
    # normal_role_abs_distance = BatchNormalization()(role_abs_distance)
    conv_role_distance = layers.Conv1D(filters=8, kernel_size=2, strides=1, padding='same')(role_distance)
    # conv_role_distance = BatchNormalization()(conv_role_distance)

    last_action = Input(shape=(1,), name='last_action')
    last_action_embedding = layers.Embedding(self.action_num, 16, name='last_action_embedding')(last_action)

    self_input = [role1_actions, role2_actions, role1_energy, role2_energy,
                  role1_x_y, role2_x_y, role1_baoqi, role2_baoqi, last_action]

    encoder_input = [role1_actions_embedding, role2_actions_embedding,
                     # normal_role_position, normal_role_distance, normal_role_abs_distance,
                     role_position, role_distance, role_abs_distance,
                     conv_role_distance, conv_role_position
                     ]
    query_input = [role1_energy_embedding, role2_energy_embedding,
                   role1_baoqi_embedding, role2_baoqi_embedding, last_action_embedding]

    return self_input, encoder_input, query_input


# 多层rnn堆叠
def build_stacked_rnn_model(self):
    self_input, encoder_input, query_input = general_input(self)
    self.network_type = 'stacked_rnn_model'

    # 分开做rnn效果不好
    # 这里使用结合的
    concatenate_layers = layers.concatenate(encoder_input)

    # 目前 512 - 1024 的效果是最好的，但数据量较大
    t = CuDNNLSTM(256, return_sequences=True)(concatenate_layers)
    t_status = CuDNNLSTM(1024)(t)
    # 双向lstm,效果不好
    # t_status = Bidirectional(CuDNNLSTM(1024))(concatenate_layers)
    q = layers.Flatten()(layers.concatenate(query_input))
    q = layers.Dense(1024)(q)
    # 使用multiply能使模型变化更加大，效果稍微好一下
    t_status = layers.multiply([t_status, q])
    # t_status = layers.add([t_status, q])
    # t_status = layers.concatenate(lstm_output)
    t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
    # 这里加bn层会造成过估计,不加的话又难以收敛。。
    t_status = BatchNormalization()(t_status)
    t_status = layers.LeakyReLU(0.05)(t_status)
    t_status = layers.Dense(256, kernel_initializer='he_uniform')(t_status)
    output = layers.LeakyReLU(0.05)(t_status)
    shared_model = Model(self_input, output)
    # 这里模型不能编译，不然后面无法扩充
    return shared_model


# 对每个元素rnn+注意力， flatten+全连接 然后两个相加
def build_rnn_attention_model(self):
    # 基于rnn的编码器
    self.network_type = 'rnn_attention_model'
    self_input, encoder_input, query_input = general_input(self)

    # 目前来看在rnn前或中间加dense层效果很差

    encoder_concatenate = layers.concatenate(encoder_input)

    # lstm返回多返回一个传动带变量，这里不需要
    values, h, _ = CuDNNLSTM(1024, return_sequences=True, return_state=True)(encoder_concatenate)
    # 这里模仿解码器的过程，将上一次的输出和hidden state 与 encoder_input合并作为query，这里输入的query远小于h，是个问题。。
    # embedding后多了一个维度尺寸，压平才能与h conact
    query = layers.Flatten()(layers.concatenate(query_input))
    query = layers.concatenate([query, h])
    c_vector, _ = BahdanauAttention(256)(query, values)

    t_status = layers.Dense(512, kernel_initializer='he_uniform')(c_vector)
    t_status = BatchNormalization()(t_status)
    t_status = layers.LeakyReLU(0.05)(t_status)
    t_status = layers.Dense(256, kernel_initializer='he_uniform')(t_status)
    output = layers.LeakyReLU(0.05)(t_status)
    shared_model = Model(self_input, output)
    return shared_model


def multi_attention_layer_normalizaiton(x):
    attention = SelfAttention(x.shape[-1])(x)
    ans = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, attention]))
    return ans


def feed_forward_network_layer_normalization(x):
    t_ans = layers.Dense(x.shape[-1], kernel_initializer='he_uniform', activation='relu')(x)
    t_ans = layers.Dense(x.shape[-1])(t_ans)
    ans = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, t_ans]))
    return ans


def build_multi_attention_model(self):
    self.network_type = 'multi_attention_model'
    self_input, encoder_input, query_input = general_input(self)

    # 目前来看在rnn前或中间加dense层效果很差

    encoder_concatenate = layers.concatenate(encoder_input)
    layers_output = []
    #  按照transformer 去掉decoder的输入，和xN的结构，因为这里不是翻译句子，不需要输入被翻译的文字，并逐个翻译
    # 后半部分加self-attention 把时间步压缩成1
    for layer in encoder_input:
        # 这里输出维度和词嵌入维度相同，输出的形状batch_size, time_steps, embedding_size,与原来一样,便于加残差
        # PositionalEncoding 第二参数要和输入的最后一维相同
        t = layers.Dense(256)(layer)
        # t = PositionalEncoding(self.input_steps, 256)(t)
        t = multi_attention_layer_normalizaiton(t)
        t = feed_forward_network_layer_normalization(t)
        t = multi_attention_layer_normalizaiton(t)
        t = feed_forward_network_layer_normalization(t)
        # 用普通attention把十个时间步压成一个
        t = layers.Flatten()(Attention()(t))
        layers_output.append(t)

    for layer in query_input:
        # pos_code = PositionalEncoding(self.input_steps, 64)(layer)

        t = multi_attention_layer_normalizaiton(layer)
        t = feed_forward_network_layer_normalization(t)

        t = multi_attention_layer_normalizaiton(t)
        t = feed_forward_network_layer_normalization(t)
        # 用普通attention把十个时间步压成一个
        t = layers.Flatten()(Attention()(t))
        layers_output.append(t)

    t_status = layers.concatenate(layers_output)

    t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
    output = layers.LeakyReLU(0.05)(t_status)

    shared_model = Model(self_input, output)
    # 这里模型不能编译，不然后面无法扩充
    return shared_model


if __name__ == '__main__':
    '''

    self.input_steps = 4
    input = layers.Input(shape=(self.input_steps,))
    input_embedding = layers.Embedding(10, 5)(input)
    output = SelfAttention(6)(input_embedding)
    self = Model(input, output)

    # 不编译也可以直接运行，不过速度慢
    self.predict(np.array([[1, 2, 3, 4]]))
    '''
    t = build_multi_attention_model(8)
