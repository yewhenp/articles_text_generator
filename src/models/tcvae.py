import tensorflow as tf
from .transformers import Encoder, create_padding_mask, create_look_ahead_mask, MultiHeadAttention
from .gpt import Decoder
from ..constants import ConfigKeys as ck


def sample_gaussian(mu, logvar):
    epsilon = tf.random.normal(tf.shape(logvar), name="epsilon")
    std = tf.exp(0.5 * logvar)
    return mu + tf.multiply(std, epsilon)


class TCVAE(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               input_vocab_size=vocab_size, rate=rate, vocab_size=vocab_size)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               target_vocab_size=vocab_size)

        self.prior_posterior_mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.prior_posterior_mha_dropout = tf.keras.layers.Dropout(rate)
        self.prior_posterior_mha_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.post_mulogvar = tf.keras.layers.Dense(d_model * 2)
        self.post_mulogvar_dropout = tf.keras.layers.Dropout(rate)
        self.post_mulogvar_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.prior_mulogvar = tf.keras.layers.Dense(d_model * 2)
        # self.prior_mulogvar_inner = tf.keras.layers.Dense(256)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training):
        inp = inputs
        tar = tf.concat([inputs[:, 1:], tf.zeros_like(inputs[:, 0])[:, tf.newaxis]], 1)

        padding_mask, look_ahead_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(inp, training, padding_mask)
        prior_posterior_attn_output, _ = self.prior_posterior_mha(enc_output, enc_output, enc_output, padding_mask)
        prior_posterior_attn_output = self.prior_posterior_mha_dropout(prior_posterior_attn_output, training=training)
        prior_posterior_attn_output = self.prior_posterior_mha_layernorm(prior_posterior_attn_output)

        post_mulogvar = self.post_mulogvar(prior_posterior_attn_output)
        post_mulogvar = self.post_mulogvar_dropout(post_mulogvar, training=training)
        post_mulogvar = self.post_mulogvar_layernorm(post_mulogvar)
        post_mu, post_logvar = tf.split(post_mulogvar, 2, axis=2)
        # prior_mulogvar = self.prior_mulogvar(self.prior_mulogvar_inner())
        # prior_mu, prior_logvar = tf.split(prior_mulogvar, 2, axis=1)
        latent_sample = sample_gaussian(post_mu, post_logvar)

        dec_output, attention_weights = self.decoder(inputs)

        final_output = self.final_layer(tf.concat([dec_output, latent_sample], axis=2))
        # final_output = self.final_layer(dec_output)
        return final_output

    def create_masks(self, inp, tar):
        padding_mask = create_padding_mask(inp)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return padding_mask, look_ahead_mask


def create_model(config):
    num_layers = config[ck.MODEL][ck.MODEL_CONFIG][ck.LAYERS]
    d_model = config[ck.MODEL][ck.MODEL_CONFIG][ck.EMBED_DIM]
    num_heads = config[ck.MODEL][ck.MODEL_CONFIG][ck.NUM_HEAD]
    dff = config[ck.MODEL][ck.MODEL_CONFIG][ck.FEED_FORWARD_DIM]
    vocab_size = config[ck.VOCAB_SIZE]

    return TCVAE(num_layers, d_model, num_heads, dff, vocab_size)
