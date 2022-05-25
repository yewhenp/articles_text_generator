import json

import tensorflow as tf
from tensorflow import keras
import numpy as np

from src.models.gpt_st_of_art.gpt2 import GPT2


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




from src.callbacks import MetricAndStatisticCallback


with open("configs/config_gpt_new.json") as file:
    config = json.load(file)

from src.dataset import WikitextDataset, FilmsDataset
train_ds = FilmsDataset(config)
test_ds = FilmsDataset(config, mode="test", vocabulary=train_ds.get_vocab())

wandb_callback = MetricAndStatisticCallback(config, train_ds, test_ds, use_wandb=False)

word_to_index = {}
for index, word in enumerate(train_ds.get_vocab()):
    word_to_index[word] = index

start_prompt = "thr film was"
start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
num_tokens_generated = 40
text_gen_callback = TextGeneratorAAA(num_tokens_generated, start_tokens, train_ds.get_vocab(), seq_len=config["max_sequence_len"])

model = GPT2(config)

model(next(iter(train_ds.get_dataset()))[0])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    decay_rate=0.999)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(
    opt, loss=loss_fn,
)  # No loss and optimization based on word embeddings from transformer block

model.summary()

model.fit(train_ds.get_dataset(), epochs=25, callbacks=[text_gen_callback, wandb_callback], validation_data=test_ds.get_dataset())


