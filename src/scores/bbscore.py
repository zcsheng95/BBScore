import argparse
from utils import compute_latent_sigma_m, compute_latent_likelihood
import numpy as np
import pickle

def main():
    parser = argparse.ArgumentParser(description='Given training text to estimate sigma and an input corpus to estimate BBScore, output is an array of BBscore with length equal to the number of articles in the input corpus')
    parser.add_argument("--latent_path", type=str, default="latent.pkl", help="path to the latent pickle file")
    parser.add_argument("--output_dir", type=str, default='bbscore_results.pkl', help = "path to save bbscore results")
    args = parser.parse_args()

    with open(args.latent_path, "rb") as file:
        training_latents = pickle.load(file)
        testing_latents = pickle.load(file)

    #generate a list of BBscore:
    # 1) Compute the diffusion coefficient list with "compute_latent_sigma_m": training_latents_sigma_m_list;
    training_latents_sigma_m = compute_latent_sigma_m(training_latents)
    # 2) Compute the mean value of training_latents_sigma_m as an approximated diffusion coefficient;
    sigma_m_trained = np.mean(training_latents_sigma_m)
    # 3) Compute the BBscore of latents with "compute_latent_likelihood" .
    train_result = compute_latent_likelihood(training_latents, sigma_train=sigma_m_trained, window_step=0)
    test_result = compute_latent_likelihood(testing_latents, sigma_train=sigma_m_trained, window_step=0)

    # results contain list of BBScore for each article
    print(f'The average BBScore for train set is {np.mean(train_result)}')
    print(f'The average BBScore for input is {np.mean(test_result)}')
    
    with open(args.output_dir, 'wb') as f:
        pickle.dump(test_result, f)

if __name__ == '__main__':
    main()

