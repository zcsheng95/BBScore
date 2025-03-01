import json
import numpy as np
import torch
from config import cfg
import torch.utils.data as data
from transformers import (
    AutoTokenizer
)

class Dataset(data.Dataset):
    split_pattern = '[SEP]'
    def __init__(self, model_name, train, config=cfg['data_params']):
        self.config = config
        self.model_name = model_name
        self.train_path = config['train_path']
        self.test_path = config['test_path']
        self.train = train
        self._load_data()
        self._set_tokenizer()
        self._process_data()


    def _load_data(self):
        if self.train:
            with open(self.train_path, 'r') as f:
                # each document is a single line
                self.data = f.readlines()
        else:
            with open(self.test_path, 'r') as f:
                self.data = f.readlines()
    def _set_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_length = self.tokenizer.model_max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.end_token = self.tokenizer.eos_token_id

        # add special token to separate sentences
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.split_pattern]})


    def _process_data(self):
        self.processed_data = []
        for doc_id in range(len(self.data)):
            # get all sentences in the document
            doc_info = []
            sentence_ct = 0
            # get all sentences in the document
            sentences = self.data[doc_id].replace(".\n", ". ").split('. ')[:-1]
            for sentence_i, sentence in enumerate(sentences):
                if not sentence:
                    continue
                # add end of sentence token
                sentence += '. '
                sentence_info = {
                    'text': sentence,
                    'sentence_id': sentence_ct,
                    'doc_id': doc_id
                }
                doc_info.append(sentence_info)
                sentence_ct += 1
            # Track total number of sentences in a document to each sentence info
            for info in doc_info:
                info['total_doc_sentences'] = sentence_ct

            # make sure sentences are greater than 5
            if len(doc_info) >= 5:
                self.processed_data.extend(doc_info)

        # print examples
        print('length: {}'.format(len(self.processed_data)))
        print("Examples: {}".format(self.processed_data[0]))

    def __len__(self):
        return len(self.processed_data)-1

    def __getitem__(self, index):
        doc_info = self.processed_data[index]
        sentence_idx = doc_info['sentence_id']
        output = {}
    
        if cfg['loss_params']['name'] == 'triplet':
            # locate sentence in the document
            # prepare triplet
            if sentence_idx < 3:
                index += (2 - sentence_idx)

            # update and make sure the sentence starts from index 2
            doc_info = self.processed_data[index]
            sentence_idx = doc_info['sentence_id']

            T = sentence_idx
            # sample random points,{t, t1, t2}
            t1, t2 = np.random.choice(T, 2, replace=False)
            if t2 < t1:
                t = t2
                t2 = t1
                t1 = t
            assert t1 < t2 and t2 < T
            y0 = self.processed_data[index - T + t1]['text']
            yt = self.processed_data[index - T + t2]['text']
            yT = self.processed_data[index]['text']

            doc_length = doc_info['total_doc_sentences']
            output = {
                'y0': y0,
                'yt': yt,
                'yT': yT,
                't1': t1,
                't2': t2,
                'T': T,
                'total_t': doc_length
            }
        elif cfg['loss_params']['loss'] == 'nll':
            raise NotImplementedError

        return output

    def tokenize_text(self, text, device):
        output = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        input_ids = output['input_ids'].squeeze(0)
        attention_mask = output['attention_mask'].squeeze(0)
        eos_input_ids = torch.tensor([[self.end_token] * input_ids.shape[0]])
        eos_attention = torch.tensor([[0] * input_ids.shape[0]])
        input_ids = torch.cat((input_ids, eos_input_ids.T), dim=1)
        attention_mask = torch.cat((attention_mask, eos_attention.T), dim=1)

        return input_ids.to(device), attention_mask.to(device)
