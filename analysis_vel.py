import os
import random
import numpy as np
from sklearn.model_selection import KFold
from create_local_data import get_test_data
from nlb_tools.evaluation import bits_per_spike
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import sys

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
log_offset = 1e-4

train_hi_spikes, train_ho_spikes, train_vel_segments, test_hi_spikes, test_ho_spikes, test_vel_segments = get_test_data(smooth_std=70, lag=120)

plt.plot(test_vel_segments[0, :500, 0])
plt.savefig('test.png')

# def fit_poisson(alpha, train_x, train_y, val_x):
#     val_pred = []
#     train_x = train_x.reshape((
#         train_x.shape[0] * train_x.shape[1], 
#         train_x.shape[2]
#     ))
#     train_y = train_y.reshape((
#         train_y.shape[0] * train_y.shape[1], 
#         train_y.shape[2]
#     ))
#     val_x = val_x.reshape((
#         val_x.shape[0] * val_x.shape[1], 
#         val_x.shape[2]
#     ))
#     train_x =  np.log(train_x + log_offset)
#     val_x =  np.log(val_x + log_offset)
#     for chan in range(train_y.shape[1]):
#         pr = PoissonRegressor(alpha=alpha, max_iter=500)
#         pr.fit(train_x, train_y[:, chan])
#         while pr.n_iter_ == pr.max_iter and pr.max_iter < 10000:
#             print(f"didn't converge - retraining {chan} with max_iter={pr.max_iter * 5}")
#             oldmax = pr.max_iter
#             del pr
#             pr = PoissonRegressor(alpha=alpha, max_iter=oldmax * 5)
#             pr.fit(train_x, train_y[:, chan])
#         val_pred.append(pr.predict(val_x))
#     val_rates_s = np.vstack(val_pred).T
#     return np.clip(val_rates_s, 1e-9, 1e20)


# train_hi_spikes, train_ho_spikes, train_vel_segments, test_hi_spikes, test_ho_spikes, test_vel_segments = get_test_data(smooth_std=70, lag=120)

# test_ho_rates = fit_poisson(0.1, train_hi_spikes, train_ho_spikes, test_hi_spikes)

# train_hi_spikes = train_hi_spikes.reshape((
#     train_hi_spikes.shape[0] * train_hi_spikes.shape[1], 
#     train_hi_spikes.shape[2]
# ))
# test_hi_spikes = test_hi_spikes.reshape((
#     test_hi_spikes.shape[0] * test_hi_spikes.shape[1], 
#     test_hi_spikes.shape[2]
# ))
# train_ho_spikes = train_ho_spikes.reshape((
#     train_ho_spikes.shape[0] * train_ho_spikes.shape[1], 
#     train_ho_spikes.shape[2]
# ))
# test_ho_spikes = test_ho_spikes.reshape((
#     test_ho_spikes.shape[0] * test_ho_spikes.shape[1], 
#     test_ho_spikes.shape[2]
# ))
# test_vel_segments = test_vel_segments.reshape((
#     test_vel_segments.shape[0] * test_vel_segments.shape[1], 
#     test_vel_segments.shape[2]
# ))
# train_vel_segments = train_vel_segments.reshape((
#     train_vel_segments.shape[0] * train_vel_segments.shape[1], 
#     train_vel_segments.shape[2]
# ))

# test_rates = np.concatenate((test_hi_spikes, test_ho_spikes), axis=-1)
# train_rates = np.concatenate((train_hi_spikes, train_ho_spikes), axis=-1)
# # test_rates = np.concatenate((test_hi_spikes, np.expand_dims(test_ho_rates, 0)), axis=-1)

# gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
# gscv.fit(train_rates, train_vel_segments)
# print(f'Decoding R2: {gscv.best_score_}')
# pred_vel = gscv.predict(test_rates)

# plt.plot(test_vel_segments[:500, 0])
# plt.savefig('test.png')

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