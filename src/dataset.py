import json
import os.path

import tensorflow as tf
from datasets import load_dataset
from keras.layers import TextVectorization

try:
    from .logs import logger
    from .constants import ConfigKeys as ck
except ImportError:
    from logs import logger
    from constants import ConfigKeys as ck


class WikitextDataset:
    def __init__(self, config, mode="train", vocabulary=None):
        self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=mode).to_pandas()
        self.dataset["length"] = self.dataset["text"].apply(lambda x: len(x))
        self.dataset = self.dataset[self.dataset["length"] > 2]
        self.dataset = self.dataset[self.dataset["text"].str.startswith(" =") == False]
        self.dataset = self.dataset.drop(["length"], axis=1)

        self.dataset = tf.convert_to_tensor(self.dataset["text"][:100])
        self.dataset = tf.data.Dataset.from_tensor_slices(self.dataset)
        logger.info("tensorflow dataset generated")

        if mode == "train":
            self.dataset = self.dataset.shuffle(buffer_size=256)
        self.dataset = self.dataset.batch(config[ck.BATCH_SIZE])

        logger.info("tensorflow dataset batched")

        if mode == "train":
            if not os.path.exists("vocab.json"):
                self.vectorize_layer = TextVectorization(
                    standardize="lower_and_strip_punctuation",
                    max_tokens=config[ck.VOCAB_SIZE] - 1,
                    output_mode="int",
                    output_sequence_length=config[ck.MAX_SEQUENCE_LEN] + 1,
                )
                logger.info("vectorize_layer created")

                self.vectorize_layer.adapt(self.dataset)
                logger.info("vectorize_layer adapted")

                self.vocab = self.vectorize_layer.get_vocabulary()
                logger.info("vectorize_layer got vocab")

                with open("vocab.json", 'w') as vocab_file:
                    json.dump(self.vocab, vocab_file)

            else:
                with open("vocab.json") as vocab_file:
                    self.vocab = json.load(vocab_file)
                self.vectorize_layer = TextVectorization(
                    standardize="lower_and_strip_punctuation",
                    max_tokens=config[ck.VOCAB_SIZE] - 1,
                    output_mode="int",
                    output_sequence_length=config[ck.MAX_SEQUENCE_LEN] + 1,
                    vocabulary=self.vocab,
                )
                logger.info("vocab loaded")

        elif mode == "test":
            if vocabulary is None:
                raise RuntimeError("vocabulary is None during test phase")

            self.vectorize_layer = TextVectorization(
                standardize="lower_and_strip_punctuation",
                max_tokens=config[ck.VOCAB_SIZE] - 1,
                output_mode="int",
                output_sequence_length=config[ck.MAX_SEQUENCE_LEN] + 1,
                vocabulary=vocabulary
            )
            logger.info("vectorize_layer created")

            self.vocab = vocabulary
            logger.info("vectorize_layer got vocab")

        def prepare_lm_inputs_labels(text):
            """
            Shift word sequences by 1 position so that the target for position (i) is
            word at position (i+1). The model will use all words up till position (i)
            to predict the next word.
            """
            text = tf.expand_dims(text, -1)
            tokenized_sentences = self.vectorize_layer(text)
            x = tokenized_sentences[:, :-1]
            y = tokenized_sentences[:, 1:]
            return x, y

        self.dataset = self.dataset.map(prepare_lm_inputs_labels)
        self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)
        logger.info("dataset ready")

    def get_dataset(self):
        return self.dataset

    def get_vocab(self):
        return self.vocab
