import json
from argparse import ArgumentParser, Namespace

import tensorflow as tf

from src.models.gpt import create_model
from src.dataset import WikitextDataset
from src.text_generator import TextGenerator
from src.callbacks import MetricAndStatisticCallback
from src.constants import ConfigKeys as ck


def main(args: Namespace):
    with open(args.config) as file:
        config = json.load(file)

    train_ds = WikitextDataset(config)
    test_ds = WikitextDataset(config, mode="test", vocabulary=train_ds.get_vocab())

    model = create_model(config)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam",
        loss=[loss_fn, None],
    )
    model.summary()

    wandb_callback = MetricAndStatisticCallback(config, train_ds, test_ds, use_wandb=True)
    text_generator = TextGenerator(config, train_ds, "After creating his last")

    model.fit(train_ds.get_dataset(),
              epochs=config[ck.EPOCHS],
              validation_data=test_ds.get_dataset(),
              callbacks=[wandb_callback])

    print(text_generator.generate_text(model))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=False, default="configs/config.json")
    args = parser.parse_args()
    main(args)
