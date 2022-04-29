import numpy as np
import tensorflow as tf

from ..constants import ConfigKeys as ck


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(embed_dim)  # (batch_size, seq_len, d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, look_ahead_mask, past=None):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, weights = self.mha1(x, x, attention_mask=look_ahead_mask, return_attention_scores=True)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

        return out3, weights


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(80, d_model)

        self.dec_layers = [
            DecoderLayer(embed_dim=d_model, num_heads=num_heads, dff=dff, rate=rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, past=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        if past is None:
            pasts = [None] * self.num_layers
        else:
            pasts = past

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        look_ahead_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)

        presents = []
        for i in range(self.num_layers):
            x, present = self.dec_layers[i](x, look_ahead_mask, past=pasts[i])
            presents.append(present)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, presents


def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


def create_model(config):
    num_layers = 2
    d_model = config[ck.MODEL][ck.MODEL_CONFIG][ck.EMBED_DIM]
    num_heads = config[ck.MODEL][ck.MODEL_CONFIG][ck.NUM_HEAD]
    dff = config[ck.MODEL][ck.MODEL_CONFIG][ck.FEED_FORWARD_DIM]
    target_vocab_size = config[ck.VOCAB_SIZE]

    decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           target_vocab_size=target_vocab_size)

    final_layer = tf.keras.layers.Dense(target_vocab_size)

    inputs = tf.keras.layers.Input(shape=(config[ck.MAX_SEQUENCE_LEN],), dtype=tf.int32, name="input")
    dec_output, weights = decoder(inputs)

    final_output = final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    model = tf.keras.Model(inputs=inputs, outputs=[final_output])
    return model
