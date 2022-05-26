import json
import os.path
import random
import string
from argparse import ArgumentParser

import pandas as pd
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
        self.dataset["text"] = pd.Series((". ".join(self.dataset["text"].tolist())).split(". "))
        self.dataset = self.dataset[self.dataset["text"].str.startswith("\n") == False]
        self.dataset["length"] = self.dataset["text"].apply(lambda x: len(x.split(" ")))
        self.dataset = self.dataset[self.dataset["length"] > 30]
        self.dataset = self.dataset.drop(["length"], axis=1)
        self.dataset.reset_index()

        self.dataset = tf.convert_to_tensor(self.dataset["text"])
        self.dataset = tf.data.Dataset.from_tensor_slices(self.dataset)
        logger.info("tensorflow dataset generated")
        self.dataset = self.dataset.batch(config[ck.BATCH_SIZE])

        logger.info("tensorflow dataset batched")

        if mode == "train":
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


class FilmsDataset:
    def __init__(self, config, mode="train", vocabulary=None):
        self.mode = mode
        filenames = []

        if mode == "train":
            directories = [
                "aclImdb/train/pos",
                "aclImdb/train/neg",
            ]
        else:
            directories = [
                "aclImdb/test/pos",
                "aclImdb/test/neg",
            ]
        for dir in directories:
            for f in os.listdir(dir):
                filenames.append(os.path.join(dir, f))

        if mode != "train":
            filenames = filenames[:250]
        print(f"{len(filenames)} files")

        # Create a dataset from text files
        text_ds = tf.data.TextLineDataset(filenames)
        text_ds = text_ds.shuffle(buffer_size=256)
        self.text_ds = text_ds.batch(config[ck.BATCH_SIZE])

        def custom_standardization(input_string):
            """ Remove html line-break tags and handle punctuation """
            lowercased = tf.strings.lower(input_string)
            stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
            return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")

        if mode == "train":
            # with open("vocab.json") as file:
            #     vocab = json.load(file)
            self.vectorize_layer = TextVectorization(
                standardize=custom_standardization,
                max_tokens=config[ck.VOCAB_SIZE] - 1,
                output_mode="int",
                output_sequence_length=config[ck.MAX_SEQUENCE_LEN] + 1,
                # vocabulary=vocab
            )
            logger.info("vectorize_layer created")

            self.vectorize_layer.adapt(self.text_ds)
            logger.info("vectorize_layer adapted")

            self.vocab = self.vectorize_layer.get_vocabulary()
            logger.info("vectorize_layer got vocab")

        elif mode == "test":
            if vocabulary is None:
                raise RuntimeError("vocabulary is None during test phase")

            self.vectorize_layer = TextVectorization(
                standardize=custom_standardization,
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

        self.text_ds = self.text_ds.map(prepare_lm_inputs_labels)
        self.text_ds = self.text_ds.prefetch(tf.data.AUTOTUNE)

    def get_dataset(self):
        return self.text_ds

    def get_vocab(self):
        return self.vocab


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=False, default="configs/config.json")
    args = parser.parse_args()

    with open(args.config) as file:
        config = json.load(file)

    train_ds = WikitextDataset(config)
    test_ds = WikitextDataset(config, mode="test", vocabulary=train_ds.get_vocab())
    y_texts = []

    for ds_entry in iter(train_ds.get_dataset()):
        y_texts.extend([" ".join([train_ds.get_vocab()[token] for token in batch]) for batch in ds_entry[0].numpy()])
    y_texts_lst = (" ".join(y_texts)).split(" ")
    print("total = ", len(y_texts_lst))
    print("unk = ", y_texts_lst.count("[UNK]") + y_texts_lst.count(""))
    print(y_texts_lst[100:120])
