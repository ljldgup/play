from tensorflow.python.keras import layers
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import concatenate, BatchNormalization, CuDNNLSTM
from tensorflow.keras import backend as K
from common.attention_related import Attention, SelfAttention, BahdanauAttention
import tensorflow as tf

# 输入配合raw_env_data_to_input使用
from common.transformer import Transformer, positional_encoding, MultiHeadAttention, point_wise_feed_forward_network


def general_input(self):
    role1_actions = Input(shape=(self.input_steps,), name='role1_actions')
    role2_actions = Input(shape=(self.input_steps,), name='role2_actions')
    # 鉴于embedding就是onehot+全连接，这里加大embedding的size
    role1_actions_embedding = layers.Embedding(512, 64, name='role1_actions_embedding')(role1_actions)
    role2_actions_embedding = layers.Embedding(512, 64, name='role2_actions_embedding')(role2_actions)

    role1_energy = Input(shape=(self.input_steps,), name='role1_energy')
    role1_energy_embedding = layers.Embedding(5, 4, name='role1_energy_embedding')(role1_energy)
    role2_energy = Input(shape=(self.input_steps,), name='role2_energy')
    role2_energy_embedding = layers.Embedding(5, 4, name='role2_energy_embedding')(role2_energy)

    role1_baoqi = Input(shape=(self.input_steps,), name='role1_baoqi')
    role1_baoqi_embedding = layers.Embedding(2, 8, name='role1_baoqi_embedding')(role1_baoqi)
    role2_baoqi = Input(shape=(self.input_steps,), name='role2_baoqi')
    role2_baoqi_embedding = layers.Embedding(2, 8, name='role2_baoqi_embedding')(role2_baoqi)

    role_position = Input(shape=(self.input_steps, 4), name='role_x_y')

    # 感觉这种环境每次都不同，小批量数据bn可能不太稳定，这里先不用
    # 这里加dense就是对最后一层坐标进行全连接，和timedistribute相同
    # 步长1 距离，步长2速度
    conv_role_position_1 = layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='same')(role_position)
    conv_role_position_2 = layers.Conv1D(filters=64, kernel_size=2, strides=1, padding='same')(role_position)
    # normal_role_distance = BatchNormalization()(role_distance)

    # actions_input = Input(shape=(self.input_steps,), name='last_action')
    actions_input = Input(shape=(self.action_steps,), name='last_action')
    actions_embedding = layers.Embedding(self.action_num, 64, name='last_action_embedding')(actions_input)

    model_input = [role1_actions, role2_actions, role1_energy, role2_energy,
                   role_position, role1_baoqi, role2_baoqi, actions_input]

    encoder_input = [role1_actions_embedding, role2_actions_embedding,
                     # normal_role_position, normal_role_distance, normal_role_abs_distance,
                     role_position, conv_role_position_1, conv_role_position_2, role1_energy_embedding,
                     role2_energy_embedding, role1_baoqi_embedding, role2_baoqi_embedding,
                     ]
    decoder_output = actions_embedding

    return model_input, encoder_input, decoder_output


# 多层rnn堆叠
def build_stacked_rnn_model(self):
    model_input, encoder_input, decoder_output = general_input(self)
    self.network_type = 'stacked_rnn_model'

    # 分开做rnn效果不好
    # 这里使用结合的
    concatenate_layers = layers.concatenate(encoder_input)

    # 目前 512 - 1024 的效果好，但数据量较大
    t = CuDNNLSTM(512, return_sequences=True)(concatenate_layers)
    t_status = CuDNNLSTM(1024)(t)
    # 双向lstm,效果不好
    # t_status = Bidirectional(CuDNNLSTM(1024))(concatenate_layers)
    decoder_lstm = CuDNNLSTM(256, return_sequences=True)(decoder_output)
    decoder_lstm = CuDNNLSTM(256)(decoder_lstm)
    t_status = layers.concatenate([decoder_lstm, t_status])
    # 不是同一内容起源的最好不要用add
    # t_status = layers.add([t_status, q])
    t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
    # 这里加bn层会造成过估计,不加的话又难以收敛。。
    t_status = BatchNormalization()(t_status)
    t_status = layers.LeakyReLU(0.05)(t_status)
    t_status = layers.Dense(256, kernel_initializer='he_uniform')(t_status)
    output = layers.LeakyReLU(0.05)(t_status)
    shared_model = Model(model_input, output)
    # 这里模型不能编译，不然后面无法扩充
    return shared_model


# 对每个元素rnn+注意力， flatten+全连接 然后两个相加
def build_rnn_attention_model(self):
    # 基于rnn的编码器
    self.network_type = 'rnn_attention_model'
    model_input, encoder_input, decoder_output = general_input(self)

    # 目前来看在rnn前或中间加dense层效果很差
    encoder_concatenate = layers.concatenate(encoder_input)

    # lstm返回多返回一个传动带变量，这里不需要
    values, h_env, _ = CuDNNLSTM(1024, return_sequences=True, return_state=True)(encoder_concatenate)
    # 这里模仿解码器的过程，将上一次的输出和hidden state 与 encoder_input合并作为query，这里输入的query远小于h，是个问题。。
    # embedding后多了一个维度尺寸，压平才能与h conact
    decoder_lstm, h_act, _ = CuDNNLSTM(128, return_sequences=True, return_state=True)(decoder_output)
    decoder_lstm = CuDNNLSTM(256)(decoder_lstm)
    h = layers.concatenate([h_act, h_act])
    c_vector, _ = BahdanauAttention(256)(h, values)
    # 这里我是多对一，同一序列只解码一次，所以直接用encoder的输出隐藏状态
    # 由于只输出一次，解码也不再用rnn，而是直接全连接
    t_status = layers.concatenate([c_vector, decoder_lstm])
    t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
    t_status = BatchNormalization()(t_status)
    t_status = layers.LeakyReLU(0.05)(t_status)
    t_status = layers.Dense(256, kernel_initializer='he_uniform')(t_status)
    output = layers.LeakyReLU(0.05)(t_status)
    shared_model = Model(model_input, output)
    return shared_model


def build_multi_attention_model(self):
    self.network_type = 'multi_attention_model'
    model_input, encoder_input, decoder_output = general_input(self)

    x = layers.concatenate(encoder_input)

    # 这里使用transformer建立模型读取非常慢，原因不明可能没有保存模型结构，载入需要做大量判断
    # 改成自己手动建立
    enc_d_model = 284
    pos_encoding = positional_encoding(self.input_steps, enc_d_model)
    # 环境嵌入尺寸324
    x *= tf.math.sqrt(tf.cast(enc_d_model, tf.float32))
    x += pos_encoding[:, :self.input_steps, :]
    x = tf.keras.layers.Dropout(0.1)(x)
    # 编码器
    for i in range(2):
        # MultiHeadAttention 里面的维度要和嵌入维度一样，因为有x + att1这一操作
        att1, _ = MultiHeadAttention(enc_d_model, 4)(x, x, x, None)
        att1 = tf.keras.layers.Dropout(0.1)(att1)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + att1)
        ffn = point_wise_feed_forward_network(enc_d_model, 1024)(x)
        ffn = tf.keras.layers.Dropout(0.1)(ffn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    enc_out = x
    # 之前解码器输出的动作
    x = decoder_output
    # 动作嵌入尺寸
    dec_d_model = 64
    x *= tf.math.sqrt(tf.cast(dec_d_model, tf.float32))
    pos_encoding = positional_encoding(self.action_steps, dec_d_model)
    x += pos_encoding[:, :self.action_steps, :]
    x = tf.keras.layers.Dropout(0.1)(x)
    # 这里因为action输入步数少而且窄，所以采用4层
    for i in range(4):
        att1, _ = MultiHeadAttention(dec_d_model, 2)(x, x, x, None)
        att1 = tf.keras.layers.Dropout(0.1)(att1)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + att1)

        # 这里将x序列长度转为enc_out的序列长度，特征数量不变
        att2, _ = MultiHeadAttention(dec_d_model, 2)(enc_out, enc_out, x, None)
        att2 = tf.keras.layers.Dropout(0.1)(att2)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + att2)

        ffn = point_wise_feed_forward_network(dec_d_model, 1024)(x)
        ffn = tf.keras.layers.Dropout(0.1)(ffn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    out = layers.Flatten()(x)

    shared_model = Model(model_input, out)
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
    # model_input, encoder_input, decoder_output = general_input(self)
