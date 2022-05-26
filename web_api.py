import json

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np

from src.models.gpt_st_of_art.gpt2 import GPT2
from src.models.gpt_st_of_art.tcvae import TCVAE2
from src.dataset import WikitextDataset, FilmsDataset
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu


def sample_from_batch(logitss, k):
    # logits, indices = tf.math.top_k(logitss, k=4, sorted=True)
    # indices = indices
    # return indices
    logits, indices = tf.math.top_k(logitss, k=k, sorted=True)
    indices = np.asarray(indices).astype("int32")
    preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
    preds = np.asarray(preds).astype("float32")
    res = np.zeros((indices.shape[0], indices.shape[1]))
    for btch_idx in range(len(indices)):
        for word_idx in range(len(indices[btch_idx])):
            res[btch_idx][word_idx] = np.random.choice(indices[btch_idx][word_idx], p=preds[btch_idx][word_idx])
    return res


class TextGeneratorFlask:
    def __init__(self, config_path="configs/config_gpt_new.json"):
        with open(config_path) as file:
            config = json.load(file)
        with open("vocab.json") as file:
            self.vocab = json.load(file)
        # self.ds_t = FilmsDataset(config)
        # self.ds = FilmsDataset(config, mode="test", vocabulary=self.ds_t.get_vocab())
        self.ds = WikitextDataset(config, mode="test", vocabulary=self.vocab)
        self.vocab = self.ds.get_vocab()
        self.max_tokens = 15
        self.k = 1
        self.seq_len = 64
        self.word_to_index = {}
        for index, word in enumerate(self.vocab):
            self.word_to_index[word] = index
        self.model = GPT2(config)
        print("model created")
        self.model(next(iter(self.ds.get_dataset()))[0])
        print("model batch got")
        self.model.load_weights(config["weights_path"])
        print("weights loaded from", config["weights_path"])
        # self.calculate_bleu()

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

    def calculate_bleu(self):
        y_pred_texts = []
        y_texts = []
        for x, y in self.ds.get_dataset():
            y_pred = self.model.predict(x)
            y_pred_ind = sample_from_batch(y_pred, self.k).astype("int")#.numpy()
            y_pred_ind[y_pred_ind >= len(self.vocab)] = 1
            y_pred_texts.extend([" ".join([self.vocab[token] for token in np.squeeze(batch)]) for batch in y_pred_ind])
            y_texts.extend([" ".join([self.vocab[token] for token in batch]) for batch in y.numpy()])

        result = self.bleu(y_texts, y_pred_texts)
        print(f"  BLEU = {result}")
        return result

    def sample_from(self, logits):
        logits = logits.numpy()
        logits[0] = 0
        logits[1] = 0
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = tf.keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        if number >= len(self.vocab):
            number = 0
        return self.vocab[number]

    def generate_text(self, input_string):
        start_tokens = [self.word_to_index.get(_, 1) for _ in input_string.split()]
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = self.seq_len - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.seq_len]
                sample_index = self.seq_len - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            # y, last_weights = self.model(x, cache=last_weights, return_cache=True)
            y = self.model(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            # [self.detokenize(_) for _ in start_tokens + tokens_generated]
            [self.detokenize(_) for _ in start_tokens]
        )
        return txt


text_generator = TextGeneratorFlask()
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/generate_text', methods=['POST'])
@cross_origin()
def process_json():
    json_request = request.json
    return jsonify(text_generator.generate_text(json_request["input_string"]))


if __name__ == '__main__':
    app.run(port=46304)
