import wandb
import tensorflow as tf

from src.models.gpt import create_model
from src.dataset import WikitextDataset
from src.text_generator import TextGenerator
from src.callbacks import WandbCustomCallback, TextGeneratorCallback
from src.metrics import BLEUCallback


if __name__ == '__main__':
    train_ds = WikitextDataset()
    test_ds = WikitextDataset(mode="test", vocabulary=train_ds.get_vocab())

    model = create_model()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        "adam",
        loss=[loss_fn, None],
    )
    model.summary()

    wandb.init(project="articles_text_generator", entity="yevpan")
    wandb_callback = WandbCustomCallback()

    text_generator = TextGenerator(train_ds, "After creating his last")
    text_generator_callback = TextGeneratorCallback(text_generator)

    bleu_callback = BLEUCallback(train_ds, test_ds)

    model.fit(train_ds.get_dataset(),
              epochs=25,
              validation_data=test_ds.get_dataset(),
              callbacks=[bleu_callback, wandb_callback])
