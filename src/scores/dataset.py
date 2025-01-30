import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional
import copy
import pandas as pd
import itertools

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock
import datasets
from tqdm import tqdm

from transformers import PreTrainedTokenizer
from transformers.utils import logging
from collections import defaultdict

from transformers import (
    GPT2Tokenizer,
)

import re

logger = logging.get_logger(__name__)


class LatentDataset(Dataset):

    def __init__(self,
                 encoder,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int,
                 special_words: list,
                 permute=False,
                 permute_size: int = 1,
                 local_n = None,
                 cache_dir: Optional[str] = None,
                 ):
        super(Dataset, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cl_model = encoder
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.examples = []
        self.raw_texts = []
        self.cl_texts = []
        self.cl_embeddings = []
        self.n_long = 0
        self.n_short = 0
        self.cpu_device = torch.device('cpu')
        self.cl_offset = 0
        self.lengths = defaultdict(list)
        self.special_words = special_words
        self.permute = permute
        self.permute_size = permute_size
        self.local_n = local_n
        self.length_lst = []
        self.shuffled_doc = ''
        self.cl_eos_str = self.special_words[-1]
        assert self.cl_eos_str == '[SEP]'
        self.cl_eos_id = self.tokenizer(self.cl_eos_str)['input_ids'][0]

        self.set_cl_tokenizer()
        start = time.time()
        self.process_dataset()
        end = time.time()
        print("Processing dataset took {}".format(end-start))

    def process_dataset(self):
        with open(self.file_path, encoding="utf-8") as f:
            docs = {}
            ct = 0
            for idx, row in enumerate(f.readlines()):
                if row.strip():
                    # Text used for embeddings.
                    row = self._clean2cltext(row)
                    # Text for GPT2
                    row = row.strip() # NOTE: remove break line
                    row = row.replace(". ", "[SEP]") # NOTE marking end of sentence, need extra trailing space to avoid Ġ
                    # replace existing token
                    row = f"{self.tokenizer.bos_token} {row} {self.tokenizer.eos_token}"
                    tokenized_text = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(row)) # tokenized row as ids
            
                    self.length_lst.append(len(tokenized_text))
                    self.did = idx
                    if len(tokenized_text) >= self.block_size:
                        self.n_long += 1
                        pass
                    else:
                        self.n_short += 1
                        example = self.tokenizer.build_inputs_with_special_tokens(tokenized_text)
                        self.examples.append(example)
                        self._get_cl_embeddings(tokenized_example=example, 
                                                gpt2_text=row,
                                                permute=self.permute,
                                                permute_size=self.permute_size,
                                                local_n=self.local_n)

                        if self.shuffled_doc == '':
                            continue
                        docs[ct] = self.shuffled_doc
                        ct = ct + 1
                        if len(self.examples) > 1422:
                            break
        self.docs = docs

        self.labels = copy.deepcopy(self.examples)
        print(f"#long text: {self.n_long}, #short text: {self.n_short}")

        

    def set_cl_tokenizer(self):
        self.cl_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.cl_tokenizer.pad_token = self.cl_tokenizer.eos_token

    def cl_tokenize(self, text, device):
        output = self.cl_tokenizer(
            text,
            padding=True,
            return_tensors='pt',
        )
        input_ids = output['input_ids'].squeeze(0)
        attention_mask = output['attention_mask'].squeeze(0)
        eos_input_ids = torch.tensor([[self.cl_tokenizer.eos_token_id]*input_ids.shape[0]])
        eos_attention = torch.tensor([[0]*input_ids.shape[0]])
        # make sure it is 2d array
        if input_ids.dim() == 1: input_ids = input_ids.view(-1, 1)
        if attention_mask.dim() == 1: attention_mask = attention_mask.view(-1, 1)
        input_ids = torch.cat((input_ids, eos_input_ids.T), dim=1)
        attention_mask = torch.cat((attention_mask, eos_attention.T), dim=1)
        return input_ids.to(device), attention_mask.to(device)


    def _clean2cltext(self, row):
        # Remove section tokens from text.
        #for tok in self.section_names:
        #    row = row.replace(tok, "")
        cl_text = row.replace(".\n", ". ")
        return cl_text

    def _get_cl_embeddings(self, tokenized_example, gpt2_text, permute, permute_size, local_n):
        return self.get_cl_embeddings(tokenized_example, gpt2_text, permute, permute_size, local_n)

    def get_end_points(self, tokenized_example):
        eos_idxs = [i-1 for i, x in enumerate(tokenized_example) if x == self.cl_eos_id]
        eos_idxs += [len(tokenized_example)]
        return eos_idxs

    def get_cl_embeddings(self, tokenized_example, gpt2_text, permute, permute_size, local_n):
        split_pattern = "[SEP]"
        cl_embeddings = []
        eos_idxs = self.get_end_points(tokenized_example)
        split_sentences = gpt2_text.split(split_pattern)
        split_sentences = [_ + split_pattern for _ in split_sentences[:-1]] + [split_sentences[-1]]
        orig_sentences = split_sentences.copy()
        if permute:
            # replace bos and eos token
            split_sentences[0] = split_sentences[0].replace(self.tokenizer.bos_token, "")
            split_sentences[len(split_sentences)-1] = split_sentences[len(split_sentences)-1].replace(self.tokenizer.eos_token, "")
            
            orig_sentences = split_sentences.copy() # debug code
            # remove empty string if any
            split_sentences = [s if s != '' else '[PAD]' for s in split_sentences]
            if local_n is not None:
                #local n is the number of windows in the document
                # window size is always 3
                if len(split_sentences) > 10:
                    pool = list(range(len(split_sentences)-2))

                    indices = []
                    # find indicies with minimum distance of 3
                    for combination in itertools.combinations(range(len(split_sentences)-2), local_n):
                        if all(combination[i] + 3 <= combination[i + 1] for i in range(0, local_n -1)):
                            indices.append(combination)

                    assert len(indices) > 0
                    idx = random.choice(indices)
                    idx = list(idx)
                    if len(idx) == 1:
                        k = idx[0]
                        selected_sents = split_sentences[k:k+3]
                        random.shuffle(selected_sents)
                        split_sentences[k:k+3] = selected_sents
                    else:
                        assert len(idx) == local_n, "incorrect number of windows"
                        for k in idx:
                            selected_sents = split_sentences[k:k+3]
                            random.shuffle(selected_sents)
                            split_sentences[k:k+3] = selected_sents

            
            else:
                # create block of sentences
                blocks_sentences = [split_sentences[i:min(i+permute_size,len(split_sentences))] for i in range(0, len(split_sentences), permute_size)]

                random.shuffle(blocks_sentences)
                # concatenate the shuffled blocks
                split_sentences[:] = [sentence for block in blocks_sentences for sentence in block]

            

            split_sentences[0] = self.tokenizer.bos_token + split_sentences[0]
            split_sentences[len(split_sentences)-1] = split_sentences[len(split_sentences)-1] + self.tokenizer.eos_token
        
        if len(eos_idxs) != len(split_sentences):
            print('eliminate: ', self.did)
            return
            
        
        assert len(eos_idxs) == len(split_sentences), f"EOS length : {len(eos_idxs)}, Split sentence length : {len(split_sentences)}, Error Document ID: {self.did}"
        shuffled_doc = ''.join(split_sentences)
        self.shuffled_doc = shuffled_doc
        
        cl_input_ids, cl_attention_mask = self.cl_tokenize(split_sentences, self.device)
        cl_feats = self.cl_model.forward(input_ids=cl_input_ids, attention_mask=cl_attention_mask) # 1, feat_size
        if torch.isnan(torch.sum(cl_feats)):
            print('Original:', orig_sentences)
            print('Shuffled:', split_sentences)
        # Align feats to the sentence length
        last_idx = 0
        for eos_idx, feat in zip(eos_idxs, cl_feats):
            cl_embeddings += [feat] * (eos_idx - last_idx)
            last_idx = eos_idx

        assert len(cl_embeddings) == len(tokenized_example)

        if self.cl_offset:
            cl_embeddings = cl_embeddings[self.cl_offset:] + [cl_embeddings[-1]] * self.cl_offset
        self.cl_embeddings.append(cl_embeddings)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.stack(self.cl_embeddings[i]).to(self.cpu_device),
                self.cl_texts[i]
                )