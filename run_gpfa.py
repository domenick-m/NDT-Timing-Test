import os
import neo
import random
import numpy as np
import quantities as pq
from itertools import product
from elephant.gpfa import GPFA
from nlb_tools.evaluation import bits_per_spike
from sklearn.model_selection import KFold
from create_local_data import get_train_data
from sklearn.linear_model import PoissonRegressor, Ridge

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
log_offset = 1e-4

def array_to_spiketrains(array, bin_size):
    """Convert B x T x N spiking array to list of list of SpikeTrains"""
    stList = []
    for trial in range(len(array)):
        trialList = []
        for channel in range(array.shape[2]):
            times = np.nonzero(array[trial, :, channel])[0]
            counts = array[trial, times, channel].astype(int)
            times = np.repeat(times, counts)
            st = neo.SpikeTrain(times*bin_size*pq.ms, t_stop=array.shape[1]*bin_size*pq.ms)
            trialList.append(st)
        stList.append(trialList)
    return stList

def fit_poisson(alpha, train_x, train_y, val_x):
    val_pred = []
    for chan in range(train_y.shape[1]):
        pr = PoissonRegressor(alpha=alpha, max_iter=500)
        pr.fit(train_x, train_y[:, chan])
        while pr.n_iter_ == pr.max_iter and pr.max_iter < 10000:
            print(f"didn't converge - retraining {chan} with max_iter={pr.max_iter * 5}")
            oldmax = pr.max_iter
            del pr
            pr = PoissonRegressor(alpha=alpha, max_iter=oldmax * 5)
            pr.fit(train_x, train_y[:, chan])
        val_pred.append(pr.predict(val_x))
    val_rates_s = np.vstack(val_pred).T
    return np.clip(val_rates_s, 1e-9, 1e20)

def fit_rectlin(train_factors_s, eval_factors_s, train_spikes_s, eval_spikes_s=None, alpha=0.0, thresh=1e-10):
    """Fit linear regression from factors to spikes, rectify, and return rate predictions"""
    train_in = train_factors_s if eval_spikes_s is None else np.vstack([train_factors_s, eval_factors_s])
    train_out = train_spikes_s if eval_spikes_s is None else np.vstack([train_spikes_s, eval_spikes_s])
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_in, train_out)
    train_rates_s = ridge.predict(train_factors_s)
    eval_rates_s = ridge.predict(eval_factors_s)
    rect_min = np.min([np.min(train_rates_s[train_rates_s > 0]), np.min(eval_rates_s[eval_rates_s > 0])])
    true_min = np.min([np.min(train_rates_s), np.min(eval_rates_s)])
    train_rates_s[train_rates_s < thresh] = thresh
    eval_rates_s[eval_rates_s < thresh] = thresh
    return train_rates_s, eval_rates_s


tmp_hi_data, _ = get_train_data(smooth_std=0)

kf = KFold(n_splits=5)

alpha1s = [0.0, 0.0001, 0.001, 0.01]
alpha2s = [0.0, 0.0001, 0.001, 0.01]

for latent_dim in [20, 30 ,40]:
    for train_index, val_index in kf.split(tmp_hi_data):
        heldin_smth_spikes, heldout_spikes = get_train_data(smooth_std=0)
        train_hi = heldin_smth_spikes[train_index]
        val_hi = heldin_smth_spikes[val_index]
        train_ho = heldout_spikes[train_index]
        val_ho = heldout_spikes[val_index]

        train_st_heldin = array_to_spiketrains(np.expand_dims(train_hi, 0), bin_size=10)
        eval_st_heldin = array_to_spiketrains(np.expand_dims(val_hi, 0), bin_size=10)

        gpfa = GPFA(bin_size=(10 * pq.ms), x_dim=int(latent_dim))
        train_factors = gpfa.fit_transform(train_st_heldin)
        eval_factors = gpfa.transform(eval_st_heldin)
        train_factors = np.stack([train_factors[i].T for i in range(len(train_factors))])
        eval_factors = np.stack([eval_factors[i].T for i in range(len(eval_factors))])

        print(train_factors.shape)
        print(eval_factors.shape)

    #     for alpha1, alpha2 in product(alpha1s, alpha2s):
    #     print(f"Evaluating alpha1={alpha1}, alpha2={alpha2}")
    #     res_list = []
    #     for n, (data, gpfa_res) in enumerate(zip(fold_data, fold_gpfa)):
    #         train_spikes_heldin, train_spikes_heldout, eval_spikes_heldin, train_st_heldin, eval_st_heldin, target_dict = data
    #         train_factors, eval_factors = gpfa_res

    #         train_spikes_heldin_s = flatten2d(train_spikes_heldin)
    #         train_spikes_heldout_s = flatten2d(train_spikes_heldout)
    #         eval_spikes_heldin_s = flatten2d(eval_spikes_heldin)
    #         train_factors_s = flatten2d(train_factors)
    #         eval_factors_s = flatten2d(eval_factors)

    #         train_rates_heldin_s, eval_rates_heldin_s = fit_rectlin(train_factors_s, eval_factors_s, train_spikes_heldin_s, eval_spikes_heldin_s, alpha=alpha1)
    #         train_rates_heldout_s, eval_rates_heldout_s = fit_poisson(train_rates_heldin_s, eval_rates_heldin_s, train_spikes_heldout_s, alpha=alpha2)

    #         train_rates_heldin = train_rates_heldin_s.reshape(train_spikes_heldin.shape)
    #         train_rates_heldout = train_rates_heldout_s.reshape(train_spikes_heldout.shape)
    #         eval_rates_heldin = eval_rates_heldin_s.reshape(eval_spikes_heldin.shape)
    #         eval_rates_heldout = eval_rates_heldout_s.reshape((eval_spikes_heldin.shape[0], eval_spikes_heldin.shape[1], train_spikes_heldout.shape[2]))

           

    #     fold_gpfa.append((train_factors, eval_factors))

    #     val_rates = fit_poisson(alpha, train_hi, train_ho, val_hi)
    #     split.append(bits_per_spike(np.expand_dims(val_rates, 1), np.expand_dims(val_ho, 1)))
    # print('alpha:',alpha,'std:',std)
    # print(np.mean(np.array(split)))
