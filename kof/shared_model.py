from tensorflow.python.keras import layers
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import concatenate, BatchNormalization, CuDNNLSTM
from tensorflow.keras import backend as K
from kof.attention_related import Attention, SelfAttention, PositionalEncoding


# 多层rnn
def build_stacked_rnn_model(model):
    model.network_type = 'stacked_rnn_model'
    role1_actions = Input(shape=(model.input_steps,), name='role1_actions')
    role2_actions = Input(shape=(model.input_steps,), name='role2_actions')
    # 鉴于embedding就是onehot+全连接，这里加大embedding的size
    role1_actions_embedding = layers.Embedding(512, 16, name='role1_actions_embedding')(role1_actions)
    role2_actions_embedding = layers.Embedding(512, 16, name='role2_actions_embedding')(role2_actions)

    role1_energy = Input(shape=(model.input_steps,), name='role1_energy')
    role1_energy_embedding = layers.Embedding(5, 2, name='role1_energy_embedding')(role1_energy)
    role2_energy = Input(shape=(model.input_steps,), name='role2_energy')
    role2_energy_embedding = layers.Embedding(5, 2, name='role2_energy_embedding')(role2_energy)

    role1_baoqi = Input(shape=(model.input_steps,), name='role1_baoqi')
    role1_baoqi_embedding = layers.Embedding(2, 1, name='role1_baoqi_embedding')(role1_baoqi)
    role2_baoqi = Input(shape=(model.input_steps,), name='role2_baoqi')
    role2_baoqi_embedding = layers.Embedding(2, 1, name='role2_baoqi_embedding')(role2_baoqi)

    role1_x_y = Input(shape=(model.input_steps, 2), name='role1_x_y')
    role2_x_y = Input(shape=(model.input_steps, 2), name='role2_x_y')

    role_position = concatenate([role1_x_y, role2_x_y])
    role_position = BatchNormalization()(role_position)
    # 这里加dense就是对最后一层坐标进行全连接，和timedistribute相同
    conv_role_position = layers.Conv1D(filters=8, kernel_size=2, strides=1, padding='same')(role_position)
    conv_role_position = BatchNormalization()(conv_role_position)
    role_distance = layers.Subtract()([role1_x_y, role2_x_y])
    role_distance = BatchNormalization()(role_distance)
    role_abs_distance = layers.Lambda(lambda x: K.abs(x))(role_distance)
    role_abs_distance = BatchNormalization()(role_abs_distance)
    conv_role_distance = layers.Conv1D(filters=8, kernel_size=2, strides=1, padding='same')(role_distance)
    conv_role_distance = BatchNormalization()(conv_role_distance)
    conv_role_abs_distance = layers.Conv1D(filters=8, kernel_size=2, strides=1, padding='same')(role_abs_distance)
    conv_role_abs_distance = BatchNormalization()(conv_role_abs_distance)
    # 使用attention模型
    important_layers = [role1_actions_embedding, role2_actions_embedding,
                        role_position, role_distance, conv_role_abs_distance,
                        conv_role_position, conv_role_distance, conv_role_abs_distance]
    unimportant_layers = [role1_energy_embedding, role2_energy_embedding,
                          role1_baoqi_embedding, role2_baoqi_embedding]
    '''
    # 分开做lstm
    lstm_output = []
    # 尝试逐渐增加宽度，而不是一次性增加宽度
    for layer in important_layers:
        t = CuDNNLSTM(64, return_sequences=True)(layer)
        t = CuDNNLSTM(128)(t)
        lstm_output.append(t)
    for layer in unimportant_layers:
        t = CuDNNLSTM(16, return_sequences=True)(layer)
        t = CuDNNLSTM(32)(t)
        lstm_output.append(t)
    '''
    # 再算一次结合的
    concatenate_layers = layers.concatenate(important_layers + unimportant_layers)

    t = CuDNNLSTM(128, return_sequences=True)(concatenate_layers)
    t = CuDNNLSTM(256, return_sequences=True)(t)
    t_status = CuDNNLSTM(512)(t)
    # lstm_output.append(t)

    # t_status = layers.concatenate(lstm_output)
    #
    t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
    t_status = BatchNormalization()(t_status)
    t_status = layers.LeakyReLU(0.05)(t_status)
    t_status = layers.Dense(256, kernel_initializer='he_uniform')(t_status)
    # t_status = BatchNormalization()(t_status)
    t_status = layers.LeakyReLU(0.05)(t_status)
    t_status = layers.Dense(128, kernel_initializer='he_uniform')(t_status)
    # t_status = BatchNormalization()(t_status)
    t_status = layers.LeakyReLU(0.05)(t_status)
    shared_model = Model([role1_actions, role2_actions, role1_energy, role2_energy,
                          role1_x_y, role2_x_y, role1_baoqi, role2_baoqi], t_status)
    # 这里模型不能编译，不然后面无法扩充
    return shared_model


# 对每个元素rnn+注意力， flatten+全连接 然后两个相加
def build_rnn_attention_model(model):
    model.network_type = 'rnn_attention_model'
    role1_actions = Input(shape=(model.input_steps,), name='role1_actions')
    role2_actions = Input(shape=(model.input_steps,), name='role2_actions')
    role1_actions_embedding = layers.Embedding(512, 16, name='role1_actions_embedding')(role1_actions)
    role2_actions_embedding = layers.Embedding(512, 16, name='role2_actions_embedding')(role2_actions)

    role1_energy = Input(shape=(model.input_steps,), name='role1_energy')
    role1_energy_embedding = layers.Embedding(5, 2, name='role1_energy_embedding')(role1_energy)
    role2_energy = Input(shape=(model.input_steps,), name='role2_energy')
    role2_energy_embedding = layers.Embedding(5, 2, name='role2_energy_embedding')(role2_energy)

    role1_baoqi = Input(shape=(model.input_steps,), name='role1_baoqi')
    role1_baoqi_embedding = layers.Embedding(2, 1, name='role1_baoqi_embedding')(role1_baoqi)
    role2_baoqi = Input(shape=(model.input_steps,), name='role2_baoqi')
    role2_baoqi_embedding = layers.Embedding(2, 1, name='role2_baoqi_embedding')(role2_baoqi)

    role1_x_y = Input(shape=(model.input_steps, 2), name='role1_x_y')
    role2_x_y = Input(shape=(model.input_steps, 2), name='role2_x_y')

    role_position = concatenate([role1_x_y, role2_x_y])
    role_position = BatchNormalization()(role_position)

    role_distance = layers.Subtract()([role1_x_y, role2_x_y])
    role_distance = BatchNormalization()(role_distance)
    role_abs_distance = layers.Lambda(lambda x: K.abs(x))(role_distance)
    role_abs_distance = BatchNormalization()(role_abs_distance)
    role_distance = layers.concatenate([role_distance, role_abs_distance])
    role_distance = layers.Dense(256)(role_distance)

    # 使用attention模型
    important_layers = [role1_actions_embedding, role2_actions_embedding,
                        role_position, role_distance]
    unimportant_layers = [role1_energy_embedding, role2_energy_embedding,
                          role1_baoqi_embedding, role2_baoqi_embedding]

    # 理论上在model attention外面包一层 rnn就是attention，这边暂时这么干
    # model attention 是自注意力，query也是自己，语言翻译中用于注意上下文语境，但这里很明显有问题
    layers_output = []
    for layer in important_layers:
        t = CuDNNLSTM(256, return_sequences=True)(layer)
        att = layers.Flatten()(Attention()(t))
        res = layers.Dense(256)(layers.Flatten()(layer))
        # Attention输出了单个时间步,也可以用squeeze
        ans = layers.Add()([att, res])
        layers_output.append(ans)

    for layer in unimportant_layers:
        t = CuDNNLSTM(64, return_sequences=True)(layer)
        att = layers.Flatten()(Attention()(t))
        res = layers.Dense(64)(layers.Flatten()(layer))
        # Attention输出了单个时间步,也可以用squeeze
        ans = layers.Add()([att, res])
        layers_output.append(ans)

    t_status = layers.concatenate(layers_output)

    t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
    t_status = BatchNormalization()(t_status)
    t_status = layers.LeakyReLU(0.05)(t_status)
    output = layers.Dense(128)(t_status)

    shared_model = Model([role1_actions, role2_actions, role1_energy, role2_energy,
                          role1_x_y, role2_x_y, role1_baoqi, role2_baoqi], output)
    # 这里模型不能编译，不然后面无法扩充
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


def build_multi_attention_model(model):
    model.network_type = 'multi_attention_model'
    role1_actions = Input(shape=(model.input_steps,), name='role1_actions')
    role2_actions = Input(shape=(model.input_steps,), name='role2_actions')
    # 鉴于embedding就是onehot+全连接，这里加大embedding的size
    # 貌似会导致nan的错误，原因不明
    role1_actions_embedding = layers.Embedding(512, 16, name='role1_actions_embedding')(role1_actions)
    role2_actions_embedding = layers.Embedding(512, 16, name='role2_actions_embedding')(role2_actions)

    role1_energy = Input(shape=(model.input_steps,), name='role1_energy')
    role1_energy_embedding = layers.Embedding(5, 2, name='role1_energy_embedding')(role1_energy)
    role2_energy = Input(shape=(model.input_steps,), name='role2_energy')
    role2_energy_embedding = layers.Embedding(5, 2, name='role2_energy_embedding')(role2_energy)

    role1_baoqi = Input(shape=(model.input_steps,), name='role1_baoqi')
    role1_baoqi_embedding = layers.Embedding(2, 2, name='role1_baoqi_embedding')(role1_baoqi)
    role2_baoqi = Input(shape=(model.input_steps,), name='role2_baoqi')
    role2_baoqi_embedding = layers.Embedding(2, 2, name='role2_baoqi_embedding')(role2_baoqi)

    role1_x_y = Input(shape=(model.input_steps, 2), name='role1_x_y')
    role2_x_y = Input(shape=(model.input_steps, 2), name='role2_x_y')

    role_position = concatenate([role1_x_y, role2_x_y])
    role_position = BatchNormalization()(role_position)
    # 这里加dense就是对最后一层坐标进行全连接，和timedistribute相同

    role_distance = layers.Subtract()([role1_x_y, role2_x_y])
    role_distance = BatchNormalization()(role_distance)
    role_abs_distance = layers.Lambda(lambda x: K.abs(x))(role_distance)
    role_abs_distance = BatchNormalization()(role_abs_distance)
    role_distance = layers.concatenate([role_distance, role_abs_distance])

    important_layers = [role1_actions_embedding, role2_actions_embedding,
                        role_position, role_distance]
    unimportant_layers = [role1_energy_embedding, role2_energy_embedding,
                          role1_baoqi_embedding, role2_baoqi_embedding]

    layers_output = []
    #  按照transformer 去掉decoder的输入，和xN的结构，因为这里不是翻译句子，不需要输入被翻译的文字，并逐个翻译
    # 后半部分加model-attention 把时间步压缩成1
    for layer in important_layers:
        # 这里输出维度和词嵌入维度相同，输出的形状batch_size, time_steps, embedding_size,与原来一样,便于加残差
        # PositionalEncoding 第二参数要和输入的最后一维相同
        t = layers.Dense(256)(layer)
        t = PositionalEncoding(model.input_steps, 256)(t)
        t = multi_attention_layer_normalizaiton(t)
        t = feed_forward_network_layer_normalization(t)
        t = multi_attention_layer_normalizaiton(t)
        t = feed_forward_network_layer_normalization(t)
        # 用普通attention把十个时间步压成一个
        t = layers.Flatten()(Attention()(t))
        layers_output.append(t)

    for layer in unimportant_layers:
        pos_code = PositionalEncoding(model.input_steps, 64)(layer)

        t = multi_attention_layer_normalizaiton(pos_code)
        t = feed_forward_network_layer_normalization(t)

        t = multi_attention_layer_normalizaiton(t)
        t = feed_forward_network_layer_normalization(t)
        # 用普通attention把十个时间步压成一个
        t = layers.Flatten()(Attention()(t))
        layers_output.append(t)

    t_status = layers.concatenate(layers_output)

    t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
    output = layers.LeakyReLU(0.05)(t_status)

    shared_model = Model([role1_actions, role2_actions, role1_energy, role2_energy,
                          role1_x_y, role2_x_y, role1_baoqi, role2_baoqi], output)
    # 这里模型不能编译，不然后面无法扩充
    return shared_model


if __name__ == '__main__':
    '''

    model.input_steps = 4
    input = layers.Input(shape=(model.input_steps,))
    input_embedding = layers.Embedding(10, 5)(input)
    output = SelfAttention(6)(input_embedding)
    model = Model(input, output)

    # 不编译也可以直接运行，不过速度慢
    model.predict(np.array([[1, 2, 3, 4]]))
    '''
    t = build_multi_attention_model(8)
