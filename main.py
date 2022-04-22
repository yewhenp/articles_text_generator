import tensorflow as tf

from src.models.gpt import create_model
from src.dataset import WikitextDataset
from src.text_generator import TextGenerator
from src.callbacks import MetricAndStatisticCallback, TextGeneratorCallback


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

    wandb_callback = MetricAndStatisticCallback(train_ds, test_ds, use_wandb=True)
    text_generator = TextGenerator(train_ds, "After creating his last")
    text_generator_callback = TextGeneratorCallback(text_generator)

    model.fit(train_ds.get_dataset(),
              epochs=25,
              validation_data=test_ds.get_dataset(),
              callbacks=[wandb_callback])

    print(text_generator.generate_text(model))
