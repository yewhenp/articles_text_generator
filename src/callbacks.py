import os

import tensorflow as tf
import numpy as np
import wandb

try:
    from .constants import *
    from .text_generator import TextGenerator
except ImportError:
    from constants import *
    from text_generator import TextGenerator


class WandbCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, all_weights_dir="weights", save_each=5):
        super().__init__()
        self.weights_dir = os.path.join(all_weights_dir)
        self.save_each = save_each
        self.best_train_loss = np.inf
        self.best_val_loss = np.inf

        if not os.path.exists(all_weights_dir):
            os.mkdir(all_weights_dir)
        if not os.path.exists(self.weights_dir):
            os.mkdir(self.weights_dir)

    def on_epoch_end(self, epoch, logs=None) -> tf.keras.Model:
        if logs is not None:
            wandb.log({metric: logs[metric] for metric in logs})
        if epoch % self.save_each == 0:
            self.model.save_weights(f"./{self.weights_dir}/w{epoch}.h5")
        if logs["loss"] < self.best_train_loss:
            self.best_train_loss = logs["loss"]
            self.model.save_weights(f"./{self.weights_dir}/train_best.h5")
        if logs["val_loss"] < self.best_val_loss:
            self.best_val_loss = logs["val_loss"]
            self.model.save_weights(f"./{self.weights_dir}/val_best.h5")


class TextGeneratorCallback(tf.keras.callbacks.Callback):
    def __init__(self, text_generator, print_every=1):
        self.text_generator: TextGenerator = text_generator
        self.print_every = print_every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every != 0:
            return
        print(f"generated text: {self.text_generator.generate_text(self.model)}")
