import os
import random
import numpy as np
from sklearn.model_selection import KFold
from create_local_data import get_train_data
from nlb_tools.evaluation import bits_per_spike
from sklearn.linear_model import PoissonRegressor

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
log_offset = 1e-4

def fit_poisson(alpha, train_x, train_y, val_x):
    val_pred = []
    train_x =  np.log(train_x + log_offset)
    val_x =  np.log(val_x + log_offset)
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


tmp_hi_data, _ = get_train_data(smooth_std=0)

kf = KFold(n_splits=5)

for std in np.linspace(30, 60, 4):
    for alpha in np.logspace(-3, 0, 4):
        split = []
        for train_index, val_index in kf.split(tmp_hi_data):
            heldin_smth_spikes, heldout_spikes = get_train_data(smooth_std=std)
            train_hi = heldin_smth_spikes[train_index]
            val_hi = heldin_smth_spikes[val_index]
            train_ho = heldout_spikes[train_index]
            val_ho = heldout_spikes[val_index]
            val_rates = fit_poisson(alpha, train_hi, train_ho, val_hi)
            split.append(bits_per_spike(np.expand_dims(val_rates, 1), np.expand_dims(val_ho, 1)))
        print('alpha:',alpha,'std:',std)
        print(np.mean(np.array(split)))

#  calculate co-bps for each val set then take average for each CV sweep. using highest average train an
#  OLE on the smooth spikes + inferred heldout rates


            
#     test0 = test[0].reshape((
#             test[0].shape[0] * test[0].shape[1], 
#             test[0].shape[2]
#         ))

#     test1 = test[1].reshape((
#             test[1].shape[0] * test[1].shape[1], 
#             test[1].shape[2]
#         ))
#     X_train, X_test = test0[train_index], test0[test_index]
#     print('train',X_train.shape)
#     print('test',X_test.shape)
#     y_train, y_test = test1[train_index], test1[test_index]

# def fit_poisson(train_factors_s, test_factors_s, train_spikes_s, test_spikes_s=None, alpha=0.0):
#     """Fit Poisson GLM from factors to spikes and return rate predictions"""
#     train_in = train_factors_s if test_spikes_s is None else np.vstack([train_factors_s, test_factors_s])
#     train_out = train_spikes_s if test_spikes_s is None else np.vstack([train_spikes_s, test_spikes_s])
#     train_pred = []
#     test_pred = []
#     for chan in range(train_out.shape[1]):
#         pr = PoissonRegressor(alpha=alpha, max_iter=500)
#         pr.fit(train_in, train_out[:, chan])
#         while pr.n_iter_ == pr.max_iter and pr.max_iter < 10000:
#             print(f"didn't converge - retraining {chan} with max_iter={pr.max_iter * 5}")
#             oldmax = pr.max_iter
#             del pr
#             pr = PoissonRegressor(alpha=alpha, max_iter=oldmax * 5)
#             pr.fit(train_in, train_out[:, chan])
#         train_pred.append(pr.predict(train_factors_s))
#         test_pred.append(pr.predict(test_factors_s))
#     train_rates_s = np.vstack(train_pred).T
#     test_rates_s = np.vstack(test_pred).T
#     return np.clip(train_rates_s, 1e-9, 1e20), np.clip(test_rates_s, 1e-9, 1e20)