import pandas as pd
import numpy as np
import re
import time
from multiprocessing import Pool
from tqdm import tqdm_notebook as tqdm
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score

from multiprocessing import Pool, cpu_count
from functools import partial

from PIL import Image
import matplotlib.pyplot as plt
from itertools import zip_longest


split_into_tokens = lambda text : re.sub(r"[^\w]", " ",  text).split()


def metric_stats(y_true, y_prob, thresholds=np.arange(0, 1.1, 0.1), metrics=(precision_score, recall_score, f1_score)):
    assert len(y_true) == len(y_prob)
    score_dict = defaultdict(list)
    score_dict['threshold'] = thresholds
    for thr in tqdm(thresholds):
        y_pred = (y_prob > thr).astype(int)
        for metric in metrics:
            score = metric(y_true, y_pred)
            score_dict[metric.__name__].append(score)
    return score_dict


def get_current_timestr(form="%Y%m%d_%H%M%S"):
    return time.strftime(form)


def normalize(text):
    '''Extraction tokens with re.
    Then transform every word to its normal form.

    Parameters
    ----------
        text (str) - text to transform
    
    Returns
    -------
        str - normalized text string splitted by space
    '''
    normal_words = [morph.parse(word)[0].normal_form for word in split_into_tokens(text)]
    return ' '.join(normal_words)


def normalize_safe(text):
    '''Safe extraction tokens with re.
    Then transform every word to its normal form.

    Parameters
    ----------
        text (str) - text to transform
    
    Returns
    -------
        str - normalized text string splitted by space
    '''
    try: 
        normal_words = [morph.parse(word)[0].normal_form for word in split_into_tokens(text)]
        return ' '.join(normal_words)
    except Exception as e:
        print(e, text)
        return ''


def par_apply(data, func, func_params={}, n_jobs=4, length=None, verbose=0):
    if n_jobs == -1:
        n_jobs = cpu_count() // 2

    if not length:
        try:
            length = len(data)
        except:
            pass
    
    applied_data = []    
    if n_jobs == 1:
        print('Sequential processing.')
        if verbose:
            data = tqdm(data, total=length)
        for d in data:
            applied = func(d, **func_params)
            applied_data.append(applied)
    else:
        print('Parallel processing.')
        with Pool(processes=n_jobs) as pool:
            if verbose:
                data = tqdm(pool.imap(partial(func, **func_params), data), total=length)
            else:
                data = pool.imap(partial(func, **func_params), data)
            for applied in data:
                applied_data.append(applied)
    return applied_data


def visualize(**images):
    # PLot images in one row 
    n = len(images)
    plt.figure(figsize=(16, 5*n))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(n, 1, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


import numpy as np

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
 
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def simple_target_encoding(trn_series, tst_series, target, min_samples_leaf=1, replace_with='median'):
    def check_type(ser, name):
        if type(trn_series) != pd.Series:
            return pd.Series(ser, name=name)
        else:
            return ser

    # def encoder_func(row):
    #     if row.count < min_samples_leaf:
    #         return replace_val
    #     else:
    #         return row.mean

    trn_series = check_type(trn_series, 'category')
    tst_series = check_type(tst_series, 'category')
    target = check_type(target, 'target')
    replace_val = target.agg(replace_with)

    temp_df = pd.concat([trn_series, target], axis=1)    
    agg_result = temp_df.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    encoder_func = lambda row : replace_val if row['count'] < min_samples_leaf else row['mean']
    mean_encode_mapper = agg_result.apply(encoder_func, axis=1).to_dict()
    ft_trn_series = trn_series.map(mean_encode_mapper)
    ft_tst_series = tst_series.map(mean_encode_mapper)
    return ft_trn_series, ft_tst_series


def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    def add_noise(series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool=True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


def show(img):
    if isinstance(img, (np.ndarray)):
        if len(img.shape) > 3:
            raise Exception(f'Image has invalid dimensions {img.shape}')
        plt.imshow(img)
    else:
        return img


def get_img(img_path, dims=(128, 128)):
    try:
        pil_img = Image.open(img_path)
    except:
        return np.zeros(dims + [3])+128
    if dims is None:
        return np.array(pil_img)
    resized_pil_img = pil_img.resize(dims)
    return np.array(resized_pil_img)


def chunker(iterable, chunk_size, fill=False, fillvalue=None):
    def chunker_nofill(iterable, chunk_size):
        if hasattr(iterable, '__getitem__'):
            for i in range(0, len(iterable), chunk_size):
                yield iterable[i : i + chunk_size]
        else:
            chunk = []
            for i in iterable:
                chunk.append(i)
                if len(chunk) == chunk_size:
                    yield chunk
                    chunk = []

    assert chunk_size > 0
    if fill:
        args = [iter(iterable)] * chunk_size
        return zip_longest(*args, fillvalue=fillvalue)
    else:
        return chunker_nofill(iterable, chunk_size)
