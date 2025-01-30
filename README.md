This repository contains code for paper "BBScore: A Brownian Bridge Based Metric for Assessing Text Coherence" accepted in AAAI 2024.

# BBScore calculation
To calculate BBScore, follow these steps:

1. Train a BB encoder to project text into a latent space.
2. During inference, pass the training set first to estimate sigma.
3. Input the text and sigma to generate BBScore.

Due to file size constraints, we provide a pre-trained [encoder](https://drive.google.com/drive/folders/1Eyvd4E_1EhPZdcDaoop6HeFUvzReqqn3?usp=sharing) using Wikisection training data with a latent dimension of 8. If you use this encoder, you can skip directly to step 2 for BBScore calculation. However, since the current encoder is domain-specific, we encourage you to train a customized encoder using your own data for optimal performance.


## 1. Training the encoder
  - create virtual environments
  ```
  conda create -n bbscore python=3.10.10
  ```
  - in the virtual environments, install necessary libraries
  ```
  pip install -r requirements.txt
  ```
  - in `src/encoder/config/config.yaml`, change the path and other settings
  - run `src/encoder/train_encoder.py` to train a BB encoder from pretrained model (i.e. GPT2)

## 2. Compute BBScore
  - run `scripts/get_latents.sh` to generate latents for the input text. **Note**: user will need to change the path to the trained encoder, the training corpus and the input text, as well as the output directory in the file.
  - Once the latents are calculated, run `src/scores/bbscore.py` and specify the latent directory and output directory to get the results.
  - The output would be a list of length $N$, $N$ is the number of lines(documents) in the input text file.
---
```
@inproceedings{sheng2024bbscore,
  title={BBScore: A Brownian Bridge Based Metric for Assessing Text Coherence},
  author={Sheng, Zhecheng and Zhang, Tianhao and Jiang, Chen and Kang, Dongyeop},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={13},
  pages={14937--14945},
  year={2024}
}

```