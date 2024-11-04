import sys
from run_fft import FFTProcessor
import numpy as np
import pandas as pd
import os
import argparse


class SpectrumData():
    """
    A wrapper class for reading and processing spectrum data
    """
    def __init__(self, filename):
        self.filename = filename
        self.spectrum_df = self.read_df()
    
    def read_df(self):
        df = pd.read_csv(self.filename)
        return df
    
    def get_dict(self):
        result = {}
        unique_sids = self.spectrum_df['sid'].unique()
        for sid in unique_sids:
            sid_df = self.spectrum_df[self.spectrum_df['sid'] == sid]
            result[sid] = {
                'freq': sid_df['freq'].values,
                'power': sid_df['power'].values
            }
        return result


def classify_pair(x_human: dict, x_model: dict, k_freq: int = 10, eps = 0.0, higher = 'model'):
    """
    Carry out the binary classification: 0 for human, 1 for model
    Args:
        x_human: dict, human spectrum data, returned by SpectrumData.get_dict()
        x_model: dict, model spectrum data, returned by SpectrumData.get_dict()
        k_freq: int, number of the first k frequency components used for classification
        eps: float, threshold for classification
        higher: str, 'model' or 'human', indicating which one has higher power value on the first k freqs of spectrum. Strictly, it should be obtained by plotting and comparing the two spectra. In the *_loop() functions below, we try both values and pick the best one.
    """
    assert x_human.keys() == x_model.keys()
    correct = 0
    for sid in x_human.keys():
        pow_human = x_human[sid]['power']
        pow_model = x_model[sid]['power']
        # If higher_spectrum == 'model'
        # Hypothesis: pow_model > pow_human for k_freq freqs
        if higher == 'model':
            if np.sum(pow_model[:k_freq]) - np.sum(pow_human[:k_freq]) > eps:
                correct += 1
        else:
            if np.sum(pow_model[:k_freq]) - np.sum(pow_human[:k_freq]) < eps:
                correct += 1
    return correct / len(x_human)


def select_k(human: dict, model: dict, higher: str):
    """
    Select the best k_freq for a single trial of classification
    """
    best_k, best_acc = None, 0.0
    for k in range(1, 51):
        acc = classify_pair(human, model, k_freq=k, higher=higher)
        if acc > best_acc:
            best_acc = acc
            best_k = k
    return best_k, best_acc


# Evaluation loop functions for each dataset
def eval_loop(dataset: str):
    """
    dataset: str, 'gpt-4', 'gpt-3.5', or 'gpt-3'
    """
    print(f'Evaluation for {dataset}')
    for genre in ['pubmed', 'writing', 'xsum']:
        print(f'Genre: {genre}')
        for est_name in ['mistral', 'gpt2xl', 'bigram']:
            human_filename = f'data/{genre}/{genre}_{dataset}.original.{est_name}.nllzs.fftnorm.txt'
            model_filename = f'data/{genre}/{genre}_{dataset}.sampled.{est_name}.nllzs.fftnorm.txt'
            if not os.path.exists(human_filename) or not os.path.exists(model_filename):
                continue
            sp_human = SpectrumData(human_filename)
            x_human = sp_human.get_dict()
            sp_model = SpectrumData(model_filename)
            x_model = sp_model.get_dict()

            best_k_1, best_acc_1 = select_k(x_human, x_model, higher='human')
            best_k_2, best_acc_2 = select_k(x_human, x_model, higher='model')
            if best_acc_1 > best_acc_2:
                best_k = best_k_1
                best_acc = best_acc_1
                higher = 'human'
            else:
                best_k = best_k_2
                best_acc = best_acc_2
                higher = 'model'
            print(f'   {genre}, {est_name}, best_k={best_k}, best_acc={best_acc:.4f}, higher={higher}')
    print()


def experiments():
    # Main experiments over three datasets
    eval_loop('gpt-4')
    eval_loop('gpt-3.5')
    eval_loop('gpt-3')


def main(args):
    sp_human = SpectrumData(args.human)
    x_human = sp_human.get_dict()
    sp_model = SpectrumData(args.model)
    x_model = sp_model.get_dict()

    best_k_1, best_acc_1 = select_k(x_human, x_model, higher='human')
    best_k_2, best_acc_2 = select_k(x_human, x_model, higher='model')
    if best_acc_1 > best_acc_2:
        best_k = best_k_1
        best_acc = best_acc_1
        higher = 'human'
    else:
        best_k = best_k_2
        best_acc = best_acc_2
        higher = 'model'
    print(f'best_k={best_k}, best_acc={best_acc:.4f}, higher={higher}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--human', type=str, help='Human spectrum data')
    parser.add_argument('--model', type=str, help='Model spectrum data')
    parser.add_argument('--experiments', action='store_true', help='Run experiments over three datasets', default=False)
    args = parser.parse_args()
    if args.experiments:
        print('Running experiments')
        experiments()
    else:
        print('Classifying a single pair of human and model spectrum data')
        assert args.human is not None, '--human must be specified'
        assert args.model is not None, '--model must be specified'
        main(args)