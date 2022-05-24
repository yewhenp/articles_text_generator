import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import numpy as np
import os
import re
import string
import random

# def causal_attention_mask(batch_size, n_dest, n_src, dtype):
#     """
#     Mask the upper half of the dot product matrix in self attention.
#     This prevents flow of information from future tokens to current token.
#     1's in the lower triangle, counting from the lower right corner.
#     """
#     i = tf.range(n_dest)[:, None]
#     j = tf.range(n_src)
#     m = i >= j - n_src + n_dest
#     mask = tf.cast(m, dtype)
#     mask = tf.reshape(mask, [1, n_dest, n_src])
#     mult = tf.concat(
#         [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
#     )
#     return tf.tile(mask, mult)
#
#
# class TransformerBlock(layers.Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
#         super(TransformerBlock, self).__init__()
#         self.att = layers.MultiHeadAttention(num_heads, embed_dim)
#         self.ffn = keras.Sequential(
#             [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
#         )
#         self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = layers.Dropout(rate)
#         self.dropout2 = layers.Dropout(rate)
#
#     def call(self, inputs):
#         input_shape = tf.shape(inputs)
#         batch_size = input_shape[0]
#         seq_len = input_shape[1]
#         causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
#         attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
#         attention_output = self.dropout1(attention_output)
#         out1 = self.layernorm1(inputs + attention_output)
#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output)
#         return self.layernorm2(out1 + ffn_output)
#
#
# class TokenAndPositionEmbedding(layers.Layer):
#     def __init__(self, maxlen, vocab_size, embed_dim):
#         super(TokenAndPositionEmbedding, self).__init__()
#         self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
#         self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
#
#     def call(self, x):
#         maxlen = tf.shape(x)[-1]
#         positions = tf.range(start=0, limit=maxlen, delta=1)
#         positions = self.pos_emb(positions)
#         x = self.token_emb(x)
#         return x + positions


# vocab_size = 30000  # Only consider the top 20k words
# maxlen = 64  # Max sequence size
# embed_dim = 128  # Embedding size for each token
# num_heads = 8  # Number of attention heads
# feed_forward_dim = 512  # Hidden layer size in feed forward network inside transformer


# def create_model():
#     inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
#     embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
#     x = embedding_layer(inputs)
#     transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
#     x = transformer_block(x)
#     outputs = layers.Dense(vocab_size)(x)
#     model = keras.Model(inputs=inputs, outputs=[outputs])
#     return model
#

# batch_size = 32

# The dataset contains each review in a separate text file
# The text files are present in four different folders
# Create a list all files
# filenames = []
# directories = [
#     "aclImdb/train/pos",
#     "aclImdb/train/neg",
#     "aclImdb/test/pos",
#     "aclImdb/test/neg",
# ]
# for dir in directories:
#     for f in os.listdir(dir):
#         filenames.append(os.path.join(dir, f))
#
# print(f"{len(filenames)} files")
#
# # Create a dataset from text files
# random.shuffle(filenames)
# text_ds = tf.data.TextLineDataset(filenames[:int(len(filenames) * 0.8)])
# text_ds = text_ds.shuffle(buffer_size=256)
# text_ds = text_ds.batch(batch_size)
#
# val_ds = tf.data.TextLineDataset(filenames[int(len(filenames) * 0.8):])
# val_ds = val_ds.shuffle(buffer_size=256)
# val_ds = val_ds.batch(batch_size)
#
#
# def custom_standardization(input_string):
#     """ Remove html line-break tags and handle punctuation """
#     lowercased = tf.strings.lower(input_string)
#     stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
#     return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")
#
#
# # Create a vectorization layer and adapt it to the text
# vectorize_layer = TextVectorization(
#     standardize=custom_standardization,
#     max_tokens=vocab_size - 1,
#     output_mode="int",
#     output_sequence_length=maxlen + 1,
# )
#vectorize_layer.adapt(text_ds)
#vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices


# def prepare_lm_inputs_labels(text):
#     """
#     Shift word sequences by 1 position so that the target for position (i) is
#     word at position (i+1). The model will use all words up till position (i)
#     to predict the next word.
#     """
#     text = tf.expand_dims(text, -1)
#     tokenized_sentences = vectorize_layer(text)
#     x = tokenized_sentences[:, :-1]
#     y = tokenized_sentences[:, 1:]
#     return x, y


#text_ds = text_ds.map(prepare_lm_inputs_labels)
#text_ds = text_ds.prefetch(tf.data.AUTOTUNE)

#val_ds = val_ds.map(prepare_lm_inputs_labels)
#val_ds = val_ds.prefetch(tf.data.AUTOTUNE)


class TextGeneratorAAA(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(
        self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1, seq_len=64
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k
        self.seq_len = seq_len

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        if number >= len(self.index_to_word):
            number = 0
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = self.seq_len - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.seq_len]
                sample_index = self.seq_len - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        print(f"generated text:\n{txt}\n")


# Tokenize starting prompt
#word_to_index = {}
#for index, word in enumerate(vocab):
#    word_to_index[word] = index

#start_prompt = "this movie is"
#start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
#num_tokens_generated = 40
#text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, vocab)


#from src.callbacks import MetricAndStatisticCallback
from src.text_generator import TextGenerator
from src.callbacks import MetricAndStatisticCallback, TextGeneratorCallback


class CallbackDSWrapper:
    def __init__(self, ds, vocab):
        self.ds = ds
        self.vocab = vocab

    def get_dataset(self):
        return self.ds

    def get_vocab(self):
        return self.vocab

with open("configs/config.json") as file:
    config = json.load(file)
#ds_wrap_train = CallbackDSWrapper(text_ds, vocab)
#ds_wrap_val = CallbackDSWrapper(val_ds, vocab)
from src.dataset import WikitextDataset, FilmsDataset
train_ds = WikitextDataset(config)
test_ds = WikitextDataset(config, mode="test", vocabulary=train_ds.get_vocab())

#text_generator = TextGenerator(config, train_ds, "the film scenario")
#text_generator_callback = TextGeneratorCallback(text_generator)
wandb_callback = MetricAndStatisticCallback({}, train_ds, test_ds, use_wandb=True)


word_to_index = {}
for index, word in enumerate(train_ds.get_vocab()):
    word_to_index[word] = index

start_prompt = "this movie is"
start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
num_tokens_generated = 40
text_gen_callback = TextGeneratorAAA(num_tokens_generated, start_tokens, train_ds.get_vocab(), seq_len=config["max_sequence_len"])


from src.models.gpt import create_model as create_model_gpt
from src.models.transformers import create_model as create_model_transformer
from src.models.tcvae import create_model as create_model_tcvae
from src.constants import ConfigKeys as ck
#from src.dataset import WikitextDataset, FilmsDataset

#with open("configs/config.json") as file:
#    config = json.load(file)

#train_ds = FilmsDataset(config)
#test_ds = FilmsDataset(config, mode="test", vocabulary=train_ds.get_vocab())

#model = create_model()
if config[ck.MODEL][ck.MODEL_TYPE] == "transformer":
    model = create_model_transformer(config)
elif config[ck.MODEL][ck.MODEL_TYPE] == "gpt":
    model = create_model_gpt(config)
elif config[ck.MODEL][ck.MODEL_TYPE] == "tcvae":
    model = create_model_tcvae(config)

model(next(iter(train_ds.get_dataset()))[0])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    decay_rate=0.96)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(
    opt, loss=loss_fn,
)  # No loss and optimization based on word embeddings from transformer block

model.summary()

model.fit(train_ds.get_dataset(), epochs=25, callbacks=[text_gen_callback, wandb_callback], validation_data=test_ds.get_dataset())


