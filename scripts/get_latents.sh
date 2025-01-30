#!/bin/bash

python ../src/scores/get_latents.py \
    --encoder src/ckpt/wiki_dim8.ckpt \
    --input data/wikisection/wikisection.test.txt \
    --train_corpus data/wikisection/wikisection.train.txt \
    --dimension 8 \
    --output output/latents 
