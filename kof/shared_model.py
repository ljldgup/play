from tensorflow.python.keras import layers
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import concatenate, BatchNormalization, CuDNNLSTM

from kof.attention_related import Attention, SelfAttention, PositionalEncoding


# 对每个元素rnn+注意力， flatten+全连接 然后两个相加
def build_rnn_attention_model(input_steps):
    role1_actions = Input(shape=(input_steps,), name='role1_actions')
    role2_actions = Input(shape=(input_steps,), name='role2_actions')
    role1_actions_embedding = layers.Embedding(512, 8, name='role1_actions_embedding')(role1_actions)
    role2_actions_embedding = layers.Embedding(512, 8, name='role2_actions_embedding')(role2_actions)

    role1_energy = Input(shape=(input_steps,), name='role1_energy')
    role1_energy_embedding = layers.Embedding(5, 1, name='role1_energy_embedding')(role1_energy)
    role2_energy = Input(shape=(input_steps,), name='role2_energy')
    role2_energy_embedding = layers.Embedding(5, 1, name='role2_energy_embedding')(role2_energy)

    role1_baoqi = Input(shape=(input_steps,), name='role1_baoqi')
    role1_baoqi_embedding = layers.Embedding(2, 1, name='role1_baoqi_embedding')(role1_baoqi)
    role2_baoqi = Input(shape=(input_steps,), name='role2_baoqi')
    role2_baoqi_embedding = layers.Embedding(2, 1, name='role2_baoqi_embedding')(role2_baoqi)

    role1_x_y = Input(shape=(input_steps, 2), name='role1_x_y')
    role2_x_y = Input(shape=(input_steps, 2), name='role2_x_y')

    role_position = concatenate([role1_x_y, role2_x_y])
    role_position = BatchNormalization()(role_position)

    role_distance = layers.Subtract()([role1_x_y, role2_x_y])
    role_distance = BatchNormalization()(role_distance)

    # 使用attention模型
    import_layers = [role1_actions_embedding, role2_actions_embedding,
                     role_position, role_distance]
    unimport_layers = [role1_energy_embedding, role2_energy_embedding,
                       role1_baoqi_embedding, role2_baoqi_embedding]

    # 理论上在self attention外面包一层 rnn就是attention，这边暂时这么干
    # self attention 是自注意力，query也是自己，语言翻译中用于注意上下文语境，但这里很明显有问题
    layers_output = []
    for layer in import_layers:
        t = CuDNNLSTM(256, return_sequences=True)(layer)
        att = layers.Flatten()(Attention()(t))
        res = layers.Dense(256)(layers.Flatten()(layer))
        # Attention输出了单个时间步,也可以用squeeze
        ans = layers.Add()([att, res])
        layers_output.append(ans)

    for layer in unimport_layers:
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
    t_ans = layers.Dense(32, kernel_initializer='he_uniform', activation='relu')(x)
    t_ans = layers.Dense(x.shape[-1])(t_ans)
    ans = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, t_ans]))
    return ans


def build_multi_attention_model(input_steps):
    role1_actions = Input(shape=(input_steps,), name='role1_actions')
    role2_actions = Input(shape=(input_steps,), name='role2_actions')
    # 鉴于embedding就是onehot+全连接，这里加大embedding的size
    role1_actions_embedding = layers.Embedding(512, 256, name='role1_actions_embedding')(role1_actions)
    role2_actions_embedding = layers.Embedding(512, 256, name='role2_actions_embedding')(role2_actions)

    role1_energy = Input(shape=(input_steps,), name='role1_energy')
    role1_energy_embedding = layers.Embedding(5, 64, name='role1_energy_embedding')(role1_energy)
    role2_energy = Input(shape=(input_steps,), name='role2_energy')
    role2_energy_embedding = layers.Embedding(5, 64, name='role2_energy_embedding')(role2_energy)

    role1_baoqi = Input(shape=(input_steps,), name='role1_baoqi')
    role1_baoqi_embedding = layers.Embedding(2, 64, name='role1_baoqi_embedding')(role1_baoqi)
    role2_baoqi = Input(shape=(input_steps,), name='role2_baoqi')
    role2_baoqi_embedding = layers.Embedding(2, 64, name='role2_baoqi_embedding')(role2_baoqi)

    role1_x_y = Input(shape=(input_steps, 2), name='role1_x_y')
    role2_x_y = Input(shape=(input_steps, 2), name='role2_x_y')

    role_position = concatenate([role1_x_y, role2_x_y])
    role_position = BatchNormalization()(role_position)
    # 这里加dense就是对最后一层坐标进行全连接，和timedistribute相同
    role_position = layers.Dense(256)(role_position)
    role_position = BatchNormalization()(role_position)

    role_distance = layers.Subtract()([role1_x_y, role2_x_y])
    role_distance = BatchNormalization()(role_distance)
    role_distance = layers.Dense(256)(role_distance)
    role_distance = BatchNormalization()(role_distance)

    import_layers = [role1_actions_embedding, role2_actions_embedding,
                     role_position, role_distance]
    unimport_layers = [role1_energy_embedding, role2_energy_embedding,
                       role1_baoqi_embedding, role2_baoqi_embedding]

    layers_output = []
    # 安装transformer 去掉decoder输入，和xN的结构
    for layer in import_layers:
        # 这里输出维度和词嵌入维度相同，输出的形状batch_size, time_steps, embedding_size,与原来一样,便于加残差
        # PositionalEncoding 第二参数要和输入的最后一维相同
        pos_code = PositionalEncoding(input_steps, 256)(layer)
        t = multi_attention_layer_normalizaiton(pos_code)
        t = feed_forward_network_layer_normalization(t)
        t = multi_attention_layer_normalizaiton(t)
        t = feed_forward_network_layer_normalization(t)
        # 用普通attention把十个时间步压成一个
        t = layers.Flatten()(Attention()(t))
        layers_output.append(t)

    for layer in unimport_layers:
        pos_code = PositionalEncoding(input_steps, 64)(layer)
        t = multi_attention_layer_normalizaiton(pos_code)
        t = feed_forward_network_layer_normalization(t)
        t = multi_attention_layer_normalizaiton(t)
        t = feed_forward_network_layer_normalization(t)

        t = layers.Flatten()(Attention()(t))
        layers_output.append(t)

    t_status = layers.concatenate(layers_output)

    t_status = layers.Dense(512, kernel_initializer='he_uniform')(t_status)
    t_status = layers.BatchNormalization()(t_status)
    t_status = layers.LeakyReLU(0.05)(t_status)

    t_status = layers.Dense(256)(t_status)
    output = layers.LeakyReLU(0.05)(t_status)

    shared_model = Model([role1_actions, role2_actions, role1_energy, role2_energy,
                          role1_x_y, role2_x_y, role1_baoqi, role2_baoqi], output)
    # 这里模型不能编译，不然后面无法扩充
    return shared_model


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
    t = build_multi_attention_model(8)
