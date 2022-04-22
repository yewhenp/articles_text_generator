import copy

import numpy as np

try:
    from .constants import ConfigKeys as ck
    from .utils import sample_from
except ImportError:
    from constants import ConfigKeys as ck
    from utils import sample_from


class TextGenerator:
    def __init__(self, config, text_ds, start_prompt, gen_size=40):
        word_to_index = {}
        for index, word in enumerate(text_ds.get_vocab()):
            word_to_index[word] = index

        self.gen_size = gen_size
        self.start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
        self.vocab = text_ds.get_vocab()
        self.config = config

    def generate_text(self, model):
        start_tokens = copy.deepcopy(self.start_tokens)
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.gen_size:
            pad_len = self.config[ck.MAX_SEQUENCE_LEN] - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.config[ck.MAX_SEQUENCE_LEN]]
                sample_index = self.config[ck.MAX_SEQUENCE_LEN] - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = model.predict(x)
            sample_token = sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join([self.vocab[token] for token in self.start_tokens + tokens_generated])
        return txt
