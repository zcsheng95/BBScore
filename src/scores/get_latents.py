import logging
import pickle
import argparse
import os
import numpy as np
import torch
import tqdm
from scipy import interpolate
from matplotlib import pyplot as plt
from transformers import (
    GPT2Tokenizer,
    AutoTokenizer
)
from utils import (
    get_checkpoint,
    get_special_tokens,
    get_dataset,
    get_all_latents
)


def main():

    parser = argparse.ArgumentParser(description='get latents from a collection of documents')

    parser.add_argument('-e', '--encoder', type=str, help='Encoder path')
    parser.add_argument('-t', '--train_corpus', type=str, help='training corpus used to estimate sigma')
    parser.add_argument('-i', '--input', type=str, help='Input text file')
    parser.add_argument('-d', '--dimension', type=int, default=8, help='Latent dimension')
    parser.add_argument('-o', '--output', type=str, help='Output directory')


    # length_dir='/home/sheng136/brownian-embedding/output/length'
    args = parser.parse_args()

    encoder_path = args.encoder
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    latent_dim = args.dimension
    # add some tokens for the specific datasets
    tokenizer = get_special_tokens(tokenizer=tokenizer)
    print("Loading model...")
    Encoder = get_checkpoint(
            latent_dim=latent_dim,
            token_size=len(tokenizer),
            filepath=encoder_path
        )# .to(cpu_device)
    Encoder.eval()

    def get_latents(data_path, train_path, encoder, is_mean=True, permute=False, permute_size=1):
    # test datasets (true)
        test_dataset = get_dataset(
            encoder = encoder,
            tokenizer=tokenizer,
            file_path=data_path,
            special_words=['[SEP]'],
            permute=permute,
            permute_size=permute_size,
        )
        train_dataset = get_dataset(
            encoder = encoder,
            tokenizer=tokenizer,
            file_path=train_path,
            special_words=['[SEP]'],
            permute=permute,
            permute_size=permute_size,
        )
        test_latents = get_all_latents(test_dataset, is_mean)
        print('Start encode training data for true sigma calculation...')
        train_latents = get_all_latents(train_dataset, is_mean)

        return train_latents, test_latents
    
    orig_path = args.input
    train_path = args.train_corpus
    truncated_file = os.path.basename(orig_path)

    print(f"File: {truncated_file}, Start Encoding...")
    train_latents, test_latents= get_latents(orig_path, train_path=train_path, encoder = Encoder, is_mean=False, permute=False)
    print('Saving...')

    # filepath_l = os.path.join(length_dir, f'{truncated_file}.pkl')
    filepath = os.path.join(args.output, f'{truncated_file}-{args.dimension}.pkl')


    with open(filepath, 'wb') as savef:
        pickle.dump(train_latents, savef)
        pickle.dump(test_latents, savef)


if __name__ == '__main__':
    main()