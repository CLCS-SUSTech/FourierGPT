# %%
import numpy as np
import sys
import os
sys.path.append('../')
from run_pwh_cls import SpectrumData
from heuristics_utils import get_roc, get_power_thresholds

import matplotlib.pyplot as plt
from sklearn.metrics import auc


# %%
def eval_auroc(genre: str, model_name: str):
    """
    genre: E.g., 'poem'
    model_name: E.g., 'ChatGPT'
    """
    print(f'Evaluation for {model_name} on {genre}')
    if genre == 'poem':
        human_filename = f'../data/{genre}_nll/poem_Human_llama3-8b-instruct.nllzs.fftnorm.txt'
        model_filename = f'../data/{genre}_nll/poem_{model_name}_{genre}_llama3-8b-instruct.nllzs.fftnorm.txt'
    else:
        human_filename = f'../data/{genre}_nll/Human_{genre}_llama3-8b-instruct.nllzs.fftnorm.txt'
        model_filename = f'../data/{genre}_nll/{model_name}_{genre}_llama3-8b-instruct.nllzs.fftnorm.txt'

    # Read spectrum data
    sp_human = SpectrumData(human_filename)
    sp_model = SpectrumData(model_filename)

    best_k = None
    best_roc_results = None
    best_auroc = 0.0
    for k in range(1, 20):
        roc_results = get_roc(sp_human, sp_model, k_threshold=k, n_intervals=20)
        # TODO: compute area under curve for ROC -- AUROC
        # and report best AUROC instead of best_acc
        FPRs = [x[1] for x in roc_results]
        TPRs = [x[0] for x in roc_results]
        auroc = auc(FPRs, TPRs)
        if auroc > best_auroc:
            best_auroc = auroc
            best_k = k
            best_roc_results = roc_results
    _, higher = get_power_thresholds(sp_human, sp_model, k_threshold=best_k, n_intervals=20)

    print(f'best_k={best_k}, best_auroc={best_auroc:.4f}, higher={higher}')
    # print(best_roc_results)
    print()


# %%
eval_auroc('poem', 'ChatGPT')
eval_auroc('poem', 'GPT3')
eval_auroc('poem', 'Llama2-70B-chat')
eval_auroc('poem', 'Olmo-7B-instruct')
eval_auroc('poem', 'Tulu2-dpo-70B')

# Evaluation for ChatGPT on poem
# best_k=5, best_auroc=0.7186, higher=model

# Evaluation for GPT3 on poem
# best_k=9, best_auroc=0.8116, higher=model

# Evaluation for Llama2-70B-chat on poem
# best_k=4, best_auroc=0.8186, higher=model

# Evaluation for Olmo-7B-instruct on poem
# best_k=19, best_auroc=0.7421, higher=human

# Evaluation for Tulu2-dpo-70B on poem
# best_k=19, best_auroc=0.7149, higher=human

# %%
eval_auroc('book', 'ChatGPT')
eval_auroc('book', 'GPT3')
eval_auroc('book', 'Llama2-70B-chat')
eval_auroc('book', 'Olmo-7B-instruct')
eval_auroc('book', 'Tulu2-dpo-70B')

# Evaluation for ChatGPT on book
# best_k=6, best_auroc=0.8311, higher=model

# Evaluation for GPT3 on book
# best_k=11, best_auroc=0.7407, higher=model

# Evaluation for Llama2-70B-chat on book
# best_k=4, best_auroc=0.7666, higher=model

# Evaluation for Olmo-7B-instruct on book
# best_k=1, best_auroc=0.6892, higher=human

# Evaluation for Tulu2-dpo-70B on book
# best_k=1, best_auroc=0.7315, higher=human

# %% 
eval_auroc('speech', 'ChatGPT')
eval_auroc('speech', 'GPT3')
eval_auroc('speech', 'Llama2-70B-chat')
eval_auroc('speech', 'Olmo-7B-instruct')
eval_auroc('speech', 'Tulu2-dpo-70B')

# Evaluation for ChatGPT on speech
# best_k=5, best_auroc=0.6837, higher=model

# Evaluation for GPT3 on speech
# best_k=16, best_auroc=0.6449, higher=model

# Evaluation for Llama2-70B-chat on speech
# best_k=5, best_auroc=0.7085, higher=model

# Evaluation for Olmo-7B-instruct on speech
# best_k=18, best_auroc=0.7372, higher=human

# Evaluation for Tulu2-dpo-70B on speech
# best_k=3, best_auroc=0.6618, higher=model