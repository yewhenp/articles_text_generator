import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

try:
    from .utils import sample_from_batch
    from .constants import *
    from .text_generator import TextGenerator
    from .dataset import WikitextDataset
except ImportError:
    from utils import sample_from_batch
    from constants import *
    from text_generator import TextGenerator
    from dataset import WikitextDataset


class BLEUCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_ds, test_ds, print_every=1):
        self.train_ds: WikitextDataset = train_ds
        self.test_ds: WikitextDataset = test_ds
        self.print_every = print_every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every != 0:
            return
        y_pred_texts = []
        y_texts = []
        for x, y in self.test_ds.get_dataset():
            y_pred, _ = self.model.predict(x)
            y_pred_ind = sample_from_batch(y_pred).numpy()
            y_pred_ind[y_pred_ind >= len(self.train_ds.get_vocab())] = 1
            y_pred_texts.extend([" ".join([self.train_ds.get_vocab()[token] for token in np.squeeze(batch)]) for batch in y_pred_ind])
            y_texts.extend([" ".join([self.train_ds.get_vocab()[token] for token in batch]) for batch in y.numpy()])

        print(f"  BLEU = {self.bleu(y_texts, y_pred_texts)}")

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



