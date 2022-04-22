import pathlib

import numpy as np
import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

try:
    from .logs import logger
except ImportError:
    from logs import logger


class WikitextDataset:
    def __init__(self):
        if not pathlib.Path("./wikitext_tokenized").exists():
            self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split='train')
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
            dataset_tokenized = self.dataset.map(self.__tokenize_list, batched=True)
            dataset_tokenized.save_to_disk("./wikitext_tokenized")
            logger.info("Tokens calculated")
        else:
            logger.info("using pretokenised")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
            dataset_tokenized = load_from_disk("./wikitext_tokenized")
            logger.info("Tokens loaded")

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="tf")
        logger.info("data_collator created")

        self.train_dataset = dataset_tokenized.remove_columns(['text', 'token_type_ids', 'attention_mask']).select(range(100)).to_pandas()
        logger.info("to pandas done")
        train_dataset_x = self.train_dataset[:-1]
        train_dataset_y = self.train_dataset[1:].rename(columns={"input_ids": "output"})
        self.train_dataset = pd.merge(train_dataset_x, train_dataset_y, left_index=True, right_index=True)
        logger.info("merge done")

        self.train_dataset = Dataset.from_pandas(self.train_dataset)
        self.train_dataset = self.train_dataset.to_tf_dataset(
            columns='input_ids',
            label_cols='output',
            shuffle=True,
            batch_size=1,
            collate_fn=self.data_collator,
        )
        logger.info("tensorflow dataset generated")

    def __tokenize_list(self, lst):
        res = self.tokenizer(lst["text"], truncation=True, padding='max_length')
        res["input_ids"] = np.asarray(res["input_ids"]).astype("int32")
        return res

    def get_dataset(self):
        return self.train_dataset


if __name__ == '__main__':
    ds = WikitextDataset()
    print(next(iter(ds.get_dataset())))
