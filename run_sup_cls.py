from run_fft import FFTProcessor
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Preprocessing
def circular(input: list, n: int = None, include_self: bool = True):
    """
    >>> circular([1, 2, 3, 4, 5])
    >>> [[1, 2, 3, 4, 5], [2, 3, 4, 5, 1], [3, 4, 5, 1, 2], [4, 5, 1, 2, 3], [5, 1, 2, 3, 4]]
    """
    if n is None:
        n = len(input) - 1
    output = []
    if include_self:
        output.append(input)
    for i in range(n):
        out = input[i+1:] + input[:i+1]
        output.append(out)
    return output

def get_circular_full(input_file: str, require_sid=True):
    fft_processor = FFTProcessor(method='fft', preprocess='logzs', value='norm', require_sid=False)
    nll_raw = fft_processor._read_data(data_file=input_file)
    circle_results = []
    for i, nll in enumerate(nll_raw):
        nll_c = circular(nll)
        nll_c = fft_processor._preprocess(nll_c)
        f, p, sids = fft_processor._fft_batch(nll_c, require_sid=True) # Note this `require_sid` is different from the function argument
        df = pd.DataFrame({'freq': np.concatenate(f), 
                           'power': np.concatenate(p), 
                           'circular_index': np.concatenate(sids)}) # The `sids` returned from `_fft_batch` means the index of each circular operation
        if require_sid: # This is the actual sequence id
            df['sid'] = i
        circle_results.append(df)
    df_circle = pd.concat(circle_results)
    return df_circle

def get_circular_mean(input_file: str, require_sid=True):
    """
    For each nll sequence, use circular to compute n spectra, then calculte its mean
    """
    fft_processor = FFTProcessor(method='fft', preprocess='logzs', value='norm', require_sid=False)
    nlls = fft_processor._read_data(data_file=input_file)
    freqs, powers, sids = [], [], []
    for i, nll in enumerate(nlls):
        nll_circle = circular(nll)
        data = fft_processor._preprocess(nll_circle)
        freq, power, _ = fft_processor._fft_batch(data, verbose=False)
        power_mean = np.mean(power, axis=0) # This is where the mean is calculated, different from get_circular_full()
        freqs.append(freq[0])
        powers.append(power_mean)
        sids.append(np.repeat(i, len(power_mean)))
    if require_sid:
        df = pd.DataFrame.from_dict({'freq': np.concatenate(freqs),
                                     'power': np.concatenate(powers),
                                     'sid': np.concatenate(sids)})
    else:
        df = pd.DataFrame.from_dict({'freq': np.concatenate(freqs),
                                'power': np.concatenate(powers)})
    return df


# Features and classification
def get_features(spectrum_data_file: str, interp_len: int = 500):
    df = pd.read_csv(spectrum_data_file)
    # If `sid` column does not exist, create it
    if 'sid' not in df.columns:
        df['sdiff']  = df['freq'] < df['freq'].shift(1, fill_value=0)
        df['sdiff'] = df['sdiff'].astype(int)
        df['sid'] = df['sdiff'].cumsum()

    features_interp = []
    for _, group in df.groupby('sid'):
        freqs = group['freq'].values
        features = group['power'].values
        new_freq = np.linspace(0, 0.5, interp_len)
        new_feat = np.interp(new_freq, freqs, features)
        features_interp.append(new_feat)

    return np.array(features_interp)


def run_classification(human_data_file: str, model_data_file: str, classifier: str = 'svm'):
    """
    Run classification on human and model data
    """
    x_human = get_features(human_data_file)
    y_human = np.zeros(x_human.shape[0])
    x_model = get_features(model_data_file)
    y_model = np.ones(x_model.shape[0])

    x = np.concatenate([x_human, x_model], axis=0)
    y = np.concatenate([y_human, y_model], axis=0)

    cls = make_pipeline(StandardScaler(),
                      SelectKBest(k=120),
    SVC(gamma='auto', kernel='rbf', C=1))
    scores = cross_val_score(cls, x, y, cv=5)

    print(f'Cross-validated acc: {scores}')
    print(f'Mean acc: {np.mean(scores)}')