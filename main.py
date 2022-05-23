import json
import os
from argparse import ArgumentParser, Namespace

import tensorflow as tf

from src.models.gpt import create_model as create_model_gpt
from src.models.transformers import create_model as create_model_transformer
from src.models.tcvae import create_model as create_model_tcvae
from src.dataset import WikitextDataset, FilmsDataset
from src.text_generator import TextGenerator
from src.callbacks import MetricAndStatisticCallback, TextGeneratorCallback
from src.constants import ConfigKeys as ck

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main(args: Namespace):
    with open(args.config) as file:
        config = json.load(file)

    train_ds = FilmsDataset(config)
    test_ds = FilmsDataset(config, mode="test", vocabulary=train_ds.get_vocab())

    if config[ck.MODEL][ck.MODEL_TYPE] == "transformer":
        model = create_model_transformer(config)
    elif config[ck.MODEL][ck.MODEL_TYPE] == "gpt":
        model = create_model_gpt(config)
    elif config[ck.MODEL][ck.MODEL_TYPE] == "tcvae":
        model = create_model_tcvae(config)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_fn(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=10000,
        decay_rate=0.99)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer="adam",
        loss=loss_fn,
    )
    model(next(iter(train_ds.get_dataset()))[0])
    model.summary()
    # They also adjusted the difficulty settings and ease of play so they could appeal to new players while retaining the essential components of the series ' gameplay
    text_generator = TextGenerator(config, train_ds, "the film scenario")
    text_generator_callback = TextGeneratorCallback(text_generator)

    if config[ck.MODE] == "train":
        wandb_callback = MetricAndStatisticCallback(config, train_ds, test_ds, use_wandb=True)
        model.fit(train_ds.get_dataset(),
                  epochs=config[ck.EPOCHS],
                  validation_data=test_ds.get_dataset(),
                  callbacks=[wandb_callback, text_generator_callback])
    elif config[ck.MODE] == "infer":
        model.load_weights("weights/val_best.h5")

    print(text_generator.generate_text(model))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=False, default="configs/config.json")
    args = parser.parse_args()
    main(args)
