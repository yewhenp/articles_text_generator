import tensorflow as tf

from .gpt2 import TokenAndPositionEmbedding, Transformer, get_tensor_shape
from ..tcvae import Encoder, create_padding_mask, create_look_ahead_mask, sample_gaussian


class TCVAE2(tf.keras.Model):
    def __init__(self, config, name=None, trainable=True, dtype=None):
        super().__init__(name=name)

        d_model = config["model"]["config"]["embed_dim"]
        num_heads = config["model"]["config"]["num_heads"]
        vocab_size = config["vocab_size"]
        maxlen = config["max_sequence_len"]
        num_layers = config["model"]["config"]["layers"]
        dff = config["model"]["config"]["dff"]
        rate = 0.1

        self.trainable = trainable
        self.embedding = TokenAndPositionEmbedding(
            maxlen=maxlen,
            vocab_size=vocab_size,
            embed_dim=d_model
        )
        self.encoder = Encoder(num_layers=num_layers // 2, d_model=d_model // 2,
                               num_heads=num_heads // 2, dff=dff // 2,
                               input_vocab_size=vocab_size, rate=0.1, maximum_position_encoding=maxlen)
        self.transformer = Transformer(config, name="transformer")
        self.prior_posterior_mha = tf.keras.layers.MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.prior_posterior_mha_dropout = tf.keras.layers.Dropout(rate)
        self.prior_posterior_mha_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.post_mulogvar = tf.keras.layers.Dense(d_model * 2, activation='tanh')
        self.post_mulogvar_dropout = tf.keras.layers.Dropout(rate)
        self.post_mulogvar_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.final_layer = tf.keras.layers.Dense(d_model)
        self.vocab_size = config["vocab_size"]

    def call(self, inputs, cache=None,
             dropout=None, attention_dropout=None,
             return_cache=False, return_logits=True, use_2d=False):
        """
        inputs: an integer tensor of shape [batch_size, seq_length] if not use_2d is False
                else a one_hot tensor of shape [batch_size, seq_length, vocab_size]
        cache: a list of dictionaries {"key": key, "value": value} of previous keys and values. it uses for generation
        use_one_hot_keys: if True it uses one hot tensors for embedding layer.
        return_cache: if True returns new keys and values alongside output. it uses for generation.
        return_logits: if True, return logits, else return last layer embedding.
        use_2d: for tpu performances: use 2D tensors for operations and return the output in 2D shape: [batch_size * seq_length, -1]
        """
        inp = inputs
        tar = tf.concat([inputs[:, 1:], tf.zeros_like(inputs[:, 0])[:, tf.newaxis]], 1)

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        enc_output = self.encoder(inp, True, enc_padding_mask)[-1]

        prior_posterior_attn_output, _ = self.prior_posterior_mha(value=enc_output, key=enc_output, query=enc_output, attention_mask=enc_padding_mask, return_attention_scores=True)
        prior_posterior_attn_output = self.prior_posterior_mha_dropout(prior_posterior_attn_output, training=True)
        prior_posterior_attn_output = self.prior_posterior_mha_layernorm(prior_posterior_attn_output)

        post_mulogvar = self.post_mulogvar(prior_posterior_attn_output)
        post_mulogvar = self.post_mulogvar_dropout(post_mulogvar, training=True)
        post_mulogvar = self.post_mulogvar_layernorm(post_mulogvar)
        post_mu, post_logvar = tf.split(post_mulogvar, 2, axis=2)
        latent_sample = sample_gaussian(post_mu, post_logvar)

        if cache is not None:
            _cache = cache[0]["key"]
            start = get_tensor_shape(_cache)[2]
        else:
            start = None
        # x = self.embedding(inputs, start)
        x = self.embedding(inputs)
        if use_2d:
            shape = get_tensor_shape(x)
            x = tf.reshape(x, [shape[0] * shape[1], shape[2]])
            shape = shape[0:2]
        else:
            shape = None
        x = self.transformer(
            inputs=x,
            cache=cache,
            dropout=dropout,
            attention_dropout=attention_dropout,
            return_cache=return_cache,
            use_2d=use_2d,
            shape=shape
        )
        if return_cache:
            x, cache = x
        x = self.final_layer(tf.concat([x, latent_sample], axis=2))
        if return_logits:
            shape = get_tensor_shape(x)
            if not use_2d:
                x = tf.reshape(x, [shape[0] * shape[1], shape[2]])
            # logits = tf.matmul(x, self.embedding.word_embedding, transpose_b=True)
            logits = tf.matmul(x, self.embedding.token_emb.embeddings, transpose_b=True)
            if not use_2d:
                logits = tf.reshape(logits, [shape[0], shape[1], self.vocab_size])
            result = logits
        else:
            result = x

        if return_cache:
            return result, cache
        else:
            return result

    def create_masks(self, inp, tar):
        enc_padding_mask = create_padding_mask(inp)
        dec_padding_mask = create_padding_mask(inp)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask
