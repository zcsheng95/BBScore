import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
from encoder.model import GPT2Encoder
import torch

from collections import defaultdict

from transformers import (
    PreTrainedTokenizer,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import torch.nn as nn

from dataset import (
    LatentDataset,
)

# Util function for calculating BBScore from latent embeddings
def compute_norm(x, t, T, a, b, sigma=1):
    mu = a + t * (b - a) / T
    var = t * (T-t) * sigma / T
    return -np.linalg.norm(x-mu)**2/2/var

def compute_sigma_m(x, t, T, a, b):
    mu = a + t * (b - a) / T
    return T * np.linalg.norm(x-mu)**2/ (2*t*(T-t))

def compute_latent_likelihood(latents, sigma_train=1, window_step=0, alpha_option=True):
    '''
    latents: a list of latents, e.g. [latent of article 1, latent of article 2, ...]

    sigma_train: an approximated diffusion coefficient, default = 1

    window_size: default = 0: no sliding window option, BBscore is computed with start and end being
        fixed to be the starting and ending point of the latent, respectively.

        Otherwise, a value >= 1, size of a sliding window step,
        e.g. window_step = 2, return the score for a triple list [i, i+2, i+4] where i ranges from 0 to latent_len - 5
    '''
    result_all = []
    for i in range(len(latents)):
        single_latent = latents[i]
        len_of_single_latent = len(single_latent)
        res_likelihood_list = []
        if window_step == 0:
            for j in range(1, len_of_single_latent-1):
                start = single_latent[0]
                end = single_latent[-1]
                temp_result = compute_norm(single_latent[j], j+1, len_of_single_latent, start, end, sigma=sigma_train)
                if alpha_option:
                    temp_result_alpha = -np.log(2 * np.pi * (j+1) * (len_of_single_latent-j-1) / len_of_single_latent * sigma_train)
                temp_result = temp_result_alpha + temp_result
                res_likelihood_list.append(temp_result)

        else:
            for j in range(len_of_single_latent-1-2*window_step):
                start = single_latent[j]
                end = single_latent[j+2*window_step]
                temp_result = compute_norm(single_latent[j+window_step], window_step+1, 2*window_step+1, start, end, sigma=sigma_train)
                if alpha_option:
                    temp_result_alpha = -np.log(2 * np.pi * window_step * (window_step+1) / (2*window_step+1) * sigma_train)
                temp_result = temp_result_alpha + temp_result
                res_likelihood_list.append(temp_result)

        '''
            If you want to use other way to define the BBScore other than mean value, work here...
        '''
        result_all.append(np.abs(np.sum(res_likelihood_list))/(len_of_single_latent -2))

    return result_all

def compute_latent_sigma_m(latents):
    sigma_m_all = []
    for i in range(len(latents)):
        single_latent = latents[i]
        start = single_latent[0]
        end = single_latent[-1]
        len_ = len(single_latent)
        sigma_m_approx = 0
        for j in range(1, len_-1):
            temp_result = compute_sigma_m(single_latent[j], j, len_, start, end)
            sigma_m_approx += temp_result
        sigma_m_all.append(sigma_m_approx/(len_-2))
    return sigma_m_all



# Util function for generate latent embeddings for text
def get_dataset(
        encoder,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        special_words: list,
        block_size=1024,
        permute=False,
        permute_size=1,
        local_n=None,
):

    dataset = LatentDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        special_words=special_words,
        block_size=block_size,
        encoder=encoder,
        permute=permute,
        permute_size=permute_size,
        local_n=local_n
        )

    return dataset




def load_encoder(filepath, latent_dim, token_size):
    model = GPT2Encoder(
        hidden_dim=128,
        latent_dim=latent_dim,
        )

    model.model.resize_token_embeddings(token_size)
    state_dict = torch.load(filepath,
                            # map_location=torch.device('cpu'), # uncomment if using cpu
                            )
    kept_keys = []
    for name, param in model.named_parameters():
        kept_keys.append(name)
    new_dict = {}

    for k, v in state_dict['state_dict'].items():
        if "model." in k:
            new_dict[k[6:]] = v
        else:
            new_dict[k] = v


    # clear up extra keys because the difference between load_pretrained and torch.load
    loaded_keys = list(new_dict.keys())
    for key in loaded_keys:
        if key not in kept_keys:
            del new_dict[key]

    model.load_state_dict(new_dict)

    for p in model.parameters():
        p.requires_grad = False

    model.eval()

    return model


def get_checkpoint(latent_dim,
                   token_size=None,
                   filepath=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_encoder(filepath,
                          latent_dim,
                          token_size=token_size
                          )
    model.to(device)
    model = model.eval()
    return model


def get_special_tokens(tokenizer):
    # NOTE loading previous tokenizer sometimes already includes the new tokens
    eos = tokenizer('[SEP]')['input_ids']
    print("Old tokenizer size: ", len(tokenizer))
    if len(eos) == 1 and eos[0] == 50257:
        print("Not adding because it's already contained")
        pass  # don't add cause it's already contained
    else:
        print("Adding tokens...")
        tokenizer.add_tokens('[SEP]')
    print("New tokenizer size: ", len(tokenizer))
    return tokenizer


def get_density(dataset):
    first_latents = []
    last_latents = []
    length = len(dataset)
    for text_i in range(length):
        first_latents.append(dataset.cl_embeddings[text_i][0].detach().cpu().numpy())
        last_latents.append(dataset.cl_embeddings[text_i][-1].detach().cpu().numpy())
    first_latents = np.array(first_latents)
    last_latents = np.array(last_latents)
    return first_latents.mean(0), first_latents.std(0), last_latents.mean(0), last_latents.std(0)


def get_all_latents(dataset, is_mean=True):
    latents = defaultdict(list)
    embedding_length = len(dataset.cl_embeddings)
    for doc in range(embedding_length):
        # iterate each document
        for sents in range(len(dataset.cl_embeddings[doc])):
            # iterate each sentence
            if is_mean:
                latents[doc].append(abs(dataset.cl_embeddings[doc][sents].detach().cpu().numpy()).mean())
            else:
                latents[doc].append(dataset.cl_embeddings[doc][sents].detach().cpu().numpy())
    return latents


def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)