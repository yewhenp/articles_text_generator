import os

import tensorflow as tf
import numpy as np
import wandb
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

try:
    from .utils import sample_from_batch
    from .text_generator import TextGenerator
    from .dataset import WikitextDataset
except ImportError:
    from utils import sample_from_batch
    from text_generator import TextGenerator
    from dataset import WikitextDataset


class MetricAndStatisticCallback(tf.keras.callbacks.Callback):
    def __init__(self, config, train_ds, test_ds, use_wandb=False, all_weights_dir="weights", save_each=5):
        super().__init__()
        self.train_ds: WikitextDataset = train_ds
        self.test_ds: WikitextDataset = test_ds
        self.use_wandb = use_wandb
        self.weights_dir = os.path.join(all_weights_dir)
        self.save_each = save_each
        self.best_train_loss = np.inf
        self.best_val_loss = np.inf

        if self.use_wandb:
            wandb.init(project="articles_text_generator", entity="yevpan", config=config)

        if not os.path.exists(all_weights_dir):
            os.mkdir(all_weights_dir)
        if not os.path.exists(self.weights_dir):
            os.mkdir(self.weights_dir)

    def on_epoch_end(self, epoch, logs=None) -> tf.keras.Model:
        bleu_score = self.calculate_bleu()

        if logs is not None and self.use_wandb:
            to_send = {metric: logs[metric] for metric in logs}
            to_send["bleu"] = bleu_score
            wandb.log(to_send)
        if epoch % self.save_each == 0:
            self.model.save_weights(f"./{self.weights_dir}/w{epoch}.h5")
        if logs["loss"] < self.best_train_loss:
            self.best_train_loss = logs["loss"]
            self.model.save_weights(f"./{self.weights_dir}/train_best.h5")
        if logs["val_loss"] < self.best_val_loss:
            self.best_val_loss = logs["val_loss"]
            self.model.save_weights(f"./{self.weights_dir}/val_best.h5")

    def calculate_bleu(self):
        y_pred_texts = []
        y_texts = []
        for x, y in self.test_ds.get_dataset():
            y_pred = self.model.predict(x)
            y_pred_ind = sample_from_batch(y_pred).numpy()
            y_pred_ind[y_pred_ind >= len(self.train_ds.get_vocab())] = 1
            y_pred_texts.extend([" ".join([self.train_ds.get_vocab()[token] for token in np.squeeze(batch)]) for batch in y_pred_ind])
            y_texts.extend([" ".join([self.train_ds.get_vocab()[token] for token in batch]) for batch in y.numpy()])

        result = self.bleu(y_texts, y_pred_texts)
        print(f"  BLEU = {result}")
        return result

    @staticmethod
    def bleu(ref, gen):
        '''
        calculate pair wise bleu score. uses nltk implementation
        Args:
            references : a list of reference sentences
            candidates : a list of candidate(generated) sentences
        Returns:
            bleu score(float)
        '''
        ref_bleu = []
        gen_bleu = []
        for l in gen:
            gen_bleu.append(l.split())
        for i, l in enumerate(ref):
            ref_bleu.append([l.split()])
        cc = SmoothingFunction()
        score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
        return score_bleu


class TextGeneratorCallback(tf.keras.callbacks.Callback):
    def __init__(self, text_generator):
        self.text_generator: TextGenerator = text_generator

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 3 == 0:
            print(f"generated text: {self.text_generator.generate_text(self.model)}")
