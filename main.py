import json
import os
from argparse import ArgumentParser, Namespace

import tensorflow as tf

# from src.models.gpt import create_model
from src.models.transformers import create_model
from src.dataset import WikitextDataset
from src.text_generator import TextGenerator
from src.callbacks import MetricAndStatisticCallback, TextGeneratorCallback
from src.constants import ConfigKeys as ck

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main(args: Namespace):
    with open(args.config) as file:
        config = json.load(file)

    train_ds = WikitextDataset(config)
    test_ds = WikitextDataset(config, mode="test", vocabulary=train_ds.get_vocab())

    model = create_model(config)
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
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
    )
    model(next(iter(train_ds.get_dataset()))[0])
    model.summary()
    # They also adjusted the difficulty settings and ease of play so they could appeal to new players while retaining the essential components of the series ' gameplay
    text_generator = TextGenerator(config, train_ds, "they also adjusted the")
    text_generator_callback = TextGeneratorCallback(text_generator)

    if config[ck.MODE] == "train":
        wandb_callback = MetricAndStatisticCallback(config, train_ds, test_ds, use_wandb=True)
        model.fit(train_ds.get_dataset(),
                  epochs=config[ck.EPOCHS],
                  validation_data=test_ds.get_dataset(),
                  callbacks=[wandb_callback, text_generator_callback])
    elif config[ck.MODE] == "infer":
        model.load_weights("weights/train_best.h5")

    print(text_generator.generate_text(model))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=False, default="configs/config.json")
    args = parser.parse_args()
    main(args)
