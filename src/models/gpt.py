import tensorflow as tf

from ..constants import ConfigKeys as ck


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(embed_dim)  # (batch_size, seq_len, d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, look_ahead_mask):
        attn1, weights = self.mha1(x, x, attention_mask=look_ahead_mask, return_attention_scores=True)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

        return out2, weights


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, sequence_len, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.sequence_len = sequence_len

        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab_size, output_dim=d_model)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=sequence_len, output_dim=d_model)

        self.dec_layers = [
            DecoderLayer(embed_dim=d_model, num_heads=num_heads, dff=dff, rate=rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        positions = tf.range(start=0, limit=self.sequence_len, delta=1)
        positions = self.pos_emb(positions)
        x += positions

        #x = self.dropout(x)

        look_ahead_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)

        presents = []
        for i in range(self.num_layers):
            x, present = self.dec_layers[i](x, look_ahead_mask)
            presents.append(present)

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
    num_layers = config[ck.MODEL][ck.MODEL_CONFIG][ck.LAYERS]
    d_model = config[ck.MODEL][ck.MODEL_CONFIG][ck.EMBED_DIM]
    num_heads = config[ck.MODEL][ck.MODEL_CONFIG][ck.NUM_HEAD]
    dff = config[ck.MODEL][ck.MODEL_CONFIG][ck.FEED_FORWARD_DIM]
    target_vocab_size = config[ck.VOCAB_SIZE]
    sequence_len = config[ck.MAX_SEQUENCE_LEN]

    decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           target_vocab_size=target_vocab_size, sequence_len=sequence_len)

    final_layer = tf.keras.layers.Dense(target_vocab_size)

    inputs = tf.keras.layers.Input(shape=(config[ck.MAX_SEQUENCE_LEN],), dtype=tf.int32, name="input")
    dec_output, weights = decoder(inputs)

    final_output = final_layer(dec_output)
    model = tf.keras.Model(inputs=inputs, outputs=final_output)
    return model
