import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()


def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random.uniform([input_dim, output_dim], minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


class GraphRecurrentNetwork:
    def __init__(self, n_node, input_dim, output_dim, dropout=0., act=tf.nn.relu, norm=False, is_train=False, num_steps=2):

        self.n_node = n_node
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.num_steps = num_steps
        self.dropout = dropout
        self.act = act
        self.norm = norm
        self.is_train = is_train
        self.var = {}

        with tf.compat.v1.variable_scope('GRNN_vars'):
            self.var['W_init'] = glorot([input_dim, output_dim], name='W_init')
            self.var['W_agg'] = glorot([output_dim, output_dim], name='W_agg')

            self.var['W_z'] = glorot([input_dim + output_dim, output_dim], name='W_z')
            self.var['W_r'] = glorot([input_dim + output_dim, output_dim], name='W_r')
            self.var['W_h'] = glorot([input_dim + output_dim, output_dim], name='W_h')

    def _gru_update(self, h_prev, agg_neighbors, inputs):
        # GRU更新逻辑
        combined = tf.concat([inputs, agg_neighbors], axis=1)
        # 更新门
        z = tf.sigmoid(tf.matmul(combined, self.var['W_z']))
        # 重置门
        r = tf.sigmoid(tf.matmul(combined, self.var['W_r']))
        # 候选状态
        combined_reset = tf.concat([inputs, r * agg_neighbors], axis=1)
        h_tilde = tf.tanh(tf.matmul(combined_reset, self.var['W_h']))
        # 新状态
        h_new = (1 - z) * h_prev + z * h_tilde
        return h_new

    def encoder(self, inputs, adj):
        # 初始隐藏状态
        h = tf.matmul(inputs, self.var['W_init'])
        if self.norm:
            h = tf.layers.batch_normalization(h, training=self.is_train)
        h = self.act(h)

        for _ in range(self.num_steps):
            # 信息聚合：聚合邻居状态
            transformed_h = tf.matmul(h, self.var['W_agg'])
            agg = tf.sparse.sparse_dense_matmul(adj, transformed_h)

            # GRU更新节点状态
            h = self._gru_update(h, agg, inputs)

            # 正则化处理
            if self.dropout > 0:
                h = tf.nn.dropout(h, rate=self.dropout)
            if self.norm:
                h = tf.layers.batch_normalization(h, training=self.is_train)
            h = self.act(h)

        return h

    def decoder(self, embed, nd):
        embed_size = embed.shape[1]
        logits = tf.matmul(embed, tf.transpose(embed))
        logits = tf.reshape(logits, [-1, 1])
        return tf.nn.relu(logits)

    def training(self, loss, lr, l2_cof):
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        train_op = opt.minimize(loss)
        return train_op
