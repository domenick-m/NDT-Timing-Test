import os
import sys
import h5py
import torch
import numpy as np
import scipy.signal as signal
import plotly.graph_objects as go
from datasets import get_dataloaders
from setup import set_device, set_seeds
from configs.default_config import get_config_from_file
from create_local_data import make_test_data
from nlb_tools.make_tensors import h5_to_dict
from nlb_tools.nwb_interface import NWBDataset
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from turtle import color
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import plotly.graph_objects as go
import numpy as np

'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                       CREATE DATA AND LOAD MODEL                       ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
smth_std = 50 #ms
lag = 120 #ms

print('Generating data...')
make_test_data(window=30, overlap=24, lag=lag, smooth_std=smth_std)
with h5py.File('/home/dmifsud/Projects/NDT-Timing-Test/data/mc_rtt_cont_24_test.h5', 'r') as h5file:
    h5dict = h5_to_dict(h5file)

if len(sys.argv) == 1 or len(sys.argv) > 2:
        print("Invalid Arguments...\n\nYou must supply a path to a '.pt' file.")
        exit()
path = sys.argv[1]
name = path[:path.rindex('/')].split('/')[-1]
config = get_config_from_file(path[:path.rindex('/')+1]+'config.yaml')

set_device(config)
device = torch.device('cuda:0')

set_seeds(config)

model = torch.load(path).to(device)
model.name = name
model.eval()

dataset = NWBDataset('/home/dmifsud/Projects/NDT-U/data/mc_rtt_train.nwb', split_heldout=True)

has_change = dataset.data.target_pos.fillna(-1000).diff(axis=0).any(axis=1) # filling NaNs with arbitrary scalar to treat as one block
change_nan = dataset.data[has_change].isna().any(axis=1)
drop_trial = (change_nan | change_nan.shift(1, fill_value=True) | change_nan.shift(-1, fill_value=True))[:-1]
change_times = dataset.data.index[has_change]
start_times = change_times[:-1][~drop_trial]
end_times = change_times[1:][~drop_trial]
target_pos = dataset.data.target_pos.loc[start_times].to_numpy().tolist()
reach_dist = dataset.data.target_pos.loc[end_times - pd.Timedelta(1, 'ms')].to_numpy() - dataset.data.target_pos.loc[start_times - pd.Timedelta(1, 'ms')].to_numpy()
reach_angle = np.arctan2(reach_dist[:, 1], reach_dist[:, 0]) / np.pi * 180
dataset.trial_info = pd.DataFrame({
    'trial_id': np.arange(len(start_times)),
    'start_time': start_times,
    'end_time': end_times,
    'target_pos': target_pos,
    'reach_dist_x': reach_dist[:, 0],
    'reach_dist_y': reach_dist[:, 1],
    'reach_angle': reach_angle,
})

dataset.resample(10)

speed = np.linalg.norm(dataset.data.finger_vel, axis=1)
dataset.data['speed'] = speed
peak_times = dataset.calculate_onset('speed', 0.05)

dataset.smooth_spk(smth_std, name=f'smth_{smth_std}', ignore_nans=True)

lag_bins = int(round(lag / dataset.bin_width))
nans = dataset.data.finger_vel.x.isna().reset_index(drop=True)

vel = dataset.data.finger_vel[~nans.to_numpy() & ~nans.shift(lag_bins, fill_value=True).to_numpy()].to_numpy()
vel_index = dataset.data.finger_vel[~nans.to_numpy() & ~nans.shift(lag_bins, fill_value=True).to_numpy()].index

spikes_hi = dataset.data.spikes[~nans.to_numpy() & ~nans.shift(-lag_bins, fill_value=True).to_numpy()].to_numpy()
spikes_ho = dataset.data.heldout_spikes[~nans.to_numpy() & ~nans.shift(-lag_bins, fill_value=True).to_numpy()].to_numpy()

print('Done!')


'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║               HELDIN RATES VS SMTH SPIKES (RANGE SLIDER)               ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
print('Generating "HELDIN RATES VS SMTH SPIKES (RANGE SLIDER)" plot...')

with torch.no_grad():
    train_rates = []
    for spikes, heldout_spikes in zip(
        torch.Tensor(h5dict['test_spikes_heldin']).to(device), torch.Tensor(h5dict['test_spikes_heldout']).to(device)
    ):
        ho_spikes = torch.zeros_like(heldout_spikes).to(device)
        spikes_new = torch.cat([spikes, ho_spikes], -1).to(device)
        output = model(spikes_new.unsqueeze(dim=0))[:, -1, :]
        train_rates.append(output.cpu())

train_rates = torch.cat(train_rates, dim=0).exp() # turn into tensor and use exponential on rates

smth_spikes = torch.Tensor(h5dict['test_hi_smth_spikes'])
heldout_smth_spikes = torch.Tensor(h5dict['test_ho_smth_spikes'])
smth_spikes = torch.cat([smth_spikes, heldout_smth_spikes], -1)

# fig = go.Figure()
# x_range=2500

# fig.add_trace(go.Scatter(y=list(train_rates[:x_range,0]), line=dict(color="#e15759"), name="NDT Rates",))
# fig.add_trace(go.Scatter(y=list(smth_spikes[:x_range,0]), line=dict(color="#4e79a7"), name="Smooth Spikes",))
# for i in range(1, 98):
#     fig.add_trace(go.Scatter(y=list(train_rates[:x_range,i]), visible=False, line=dict(color="#e15759"), name="NDT Rates",))
#     fig.add_trace(go.Scatter(y=list(smth_spikes[:x_range,i]), visible=False, line=dict(color="#4e79a7"), name="Smooth Spikes",))

# fig.update_layout(
#     xaxis=dict(
#         rangeselector=dict(),
#         rangeslider=dict(visible=True)
#     )
# )

# buttons = []
# for i in range(98):
#     vis_list = [False for i in range(196)]
#     vis_list[i*2] = True
#     vis_list[i*2+1] = True
#     buttons.append(dict(
#         method='restyle',
#         label=f'ch {i+1}',
#         visible=True,
#         args=[{'visible':vis_list,}]
#     ))
           
# # specify updatemenu        
# um = [{
#     'buttons':buttons, 
#     'direction': 'down',
#     'pad': {"r": 0, "t": 0},
#     'showactive':True,
#     'x':0.0,
#     'xanchor':"left",
#     'y':1.075,
#     'yanchor':"bottom" 
# }]
# fig.update_layout(updatemenus=um)

# fig['layout']['xaxis'].update(range=['0', '300'])

# layout = go.Layout(
#     margin=go.layout.Margin(
#         l=0, #left margin
#         r=0, #right margin
#         b=0, #bottom margin
#         t=0  #top margin
#     )
# )
# fig.update_layout(layout)

# fig.update_xaxes(
#     ticktext=[f'{int(i/100)}s' for i in range(0, x_range, 100)],
#     tickvals=[i for i in range(0, x_range, 100)],
# )

# fig.update_layout(legend=dict(
#     yanchor="bottom",
#     y=1.035,
#     xanchor="right",
#     x=1.00
# ))

# if not os.path.isdir(f"plots/{name}"): os.makedirs(f"plots/{name}")
# fig.write_html(f"plots/{name}/spk_vs_rates_heldin_slider.html")
# print("Done!")


# '''
#    ╔════════════════════════════════════════════════════════════════════════╗
#    ║              HELDOUT RATES VS SMTH SPIKES (RANGE SLIDER)               ║
#    ╚════════════════════════════════════════════════════════════════════════╝
# '''
# print('Generating "HELDOUT RATES VS SMTH SPIKES (RANGE SLIDER)" plot...')

# fig = go.Figure()
# x_range=2500

# fig.add_trace(go.Scatter(y=list(train_rates[:x_range,98]), line=dict(color="#e15759"), name="NDT Rates",))
# fig.add_trace(go.Scatter(y=list(smth_spikes[:x_range,98]), line=dict(color="#4e79a7"), name="Smooth Spikes",))
# for i in range(99, 130):
#     fig.add_trace(go.Scatter(y=list(train_rates[:x_range,i]), visible=False, line=dict(color="#e15759"), name="NDT Rates",))
#     fig.add_trace(go.Scatter(y=list(smth_spikes[:x_range,i]), visible=False, line=dict(color="#4e79a7"), name="Smooth Spikes",))

# fig.update_layout(
#     xaxis=dict(
#         rangeselector=dict(),
#         rangeslider=dict(visible=True)
#     )
# )

# buttons = []
# for i in range(98, 130):
#     vis_list = [False for i in range(64)]
#     vis_list[(i-98)*2] = True
#     vis_list[(i-98)*2+1] = True
#     buttons.append(dict(
#         method='restyle',
#         label=f'ch {i+1}',
#         visible=True,
#         args=[{'visible':vis_list,}]
#     ))
           
# # specify updatemenu        
# um = [{
#     'buttons':buttons, 
#     'direction': 'down',
#     'pad': {"r": 0, "t": 0},
#     'showactive':True,
#     'x':0.0,
#     'xanchor':"left",
#     'y':1.075,
#     'yanchor':"bottom" 
# }]
# fig.update_layout(updatemenus=um)

# fig['layout']['xaxis'].update(range=['0', '300'])

# layout = go.Layout(
#     margin=go.layout.Margin(
#         l=0, #left margin
#         r=0, #right margin
#         b=0, #bottom margin
#         t=0  #top margin
#     )
# )
# fig.update_layout(layout)

# fig.update_xaxes(
#     ticktext=[f'{int(i/100)}s' for i in range(0, x_range, 100)],
#     tickvals=[i for i in range(0, x_range, 100)],
# )

# fig.update_layout(legend=dict(
#     yanchor="bottom",
#     y=1.035,
#     xanchor="right",
#     x=1.00
# ))

# fig.write_html(f"plots/{name}/spk_vs_rates_heldout_slider.html")
# print("Done!")


# '''
#    ╔════════════════════════════════════════════════════════════════════════╗
#    ║               HELDIN RATES VS SMTH SPIKES (ALL CHANNELS)               ║
#    ╚════════════════════════════════════════════════════════════════════════╝
# '''
# print('Generating "HELDIN RATES VS SMTH SPIKES (ALL CHANNELS)" plot...')

# def rates_string(neuron):
#     array_string = 'y: ['
#     for i in train_rates[:300,neuron]:
#         array_string += str(i.item())+','
#     array_string += '],'
#     return array_string

# def ss_string(neuron):
#     array_string = 'y: ['
#     for i in smth_spikes[:300,neuron]:
#         array_string += str(i.item())+','
#     array_string += '],'
#     return array_string

# with open(f"plots/{name}/spk_vs_rates_heldin_all.html", "w") as f:
#     f.write('<!DOCTYPE html><html lang="en" ><head><meta charset="UTF-8"><title>NDT Heldin Rates</title></head><body><!-- partial:index.partial.html --><div id="legend" style="height: 50px"></div><div style="height:450px; overflow-y: auto"><div id="plot" style="height:8000px"></div></div><div id="xaxis" style="height: 60px"></div><!-- partial --><script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.3.1/plotly.min.js"></script><script  src="./spk_vs_rates_heldin_all.js"></script></body></html>')

# with open(f"plots/{name}/spk_vs_rates_heldin_all.js", "w") as f:
#     names = []
#     for i in range(98):
#         names.append(f'trace{i+1}')
#         names.append(f'trace{i+1}r')
#         str_to_write = f'var trace{i+1} = {{'
#         str_to_write += ss_string(i)
#         str_to_write += f"marker: {{color: '#4e79a7'}},name: 'Smoothed Spikes',yaxis: 'y{i+1}',type: 'line',"
#         if i != 0:
#             str_to_write += "showlegend: false,"
#         str_to_write += f'}};\nvar trace{i+1}r = {{'
#         str_to_write += rates_string(i)
#         str_to_write += f"marker: {{color: '#e15759'}},name: 'NDT Rates',yaxis: 'y{i+1}',type: 'line',"
#         if i != 0:
#             str_to_write += "showlegend: false,"
#         str_to_write +='};\n'
#         f.write(str_to_write)
#     names_str = 'data = ['
#     for i in names:
#         names_str += f"{i}, "
#     names_str += ']'
#     f.write(names_str+f'\n')
#     f.write(f'let bottomTraces = [{{ mode: "scatter" }}];\nlet bottomLayout = {{yaxis: {{ tickmode: "array", tickvals: [], fixedrange: true }},xaxis: {{tickmode: "array",tickvals: [0, 33, 66, 100],ticktext: ["0s", "1s", "2s", "3s"],range: [0, 100],domain: [0.0, 1.0],fixedrange: true}},margin: {{ l: 25, t: 0 , r: 40}},}};\nvar config = {{responsive: true}};\nPlotly.react("plot",data,{{xaxis: {{visible: false, fixedrange: true}},grid: {{rows: 98, columns: 1}},')
#     axis_labels = f"\nyaxis: {{title: {{text: 'ch 1',}}, showticklabels: false, fixedrange: true}},\n"
#     for i in range(2,99):
#         axis_labels += f"yaxis{i}: {{title: {{text: 'ch {i}',}}, showticklabels: false, fixedrange: true}},\n"
#     f.write(axis_labels)
#     f.write('margin: { l: 25, t: 25, b: 0 , r: 25},showlegend: false,},config);\nPlotly.react("xaxis", bottomTraces, bottomLayout, { displayModeBar: false, responsive: true });\ndata = [{y: [null],name: "Smooth Spikes",mode: "lines",marker: {color: "#4e79a7"},},{y: [null],name: "NDT Rates",mode: "lines",marker: {color: "#e15759"},}];\nlet newLayout = {yaxis: { visible: false},xaxis: { visible: false},margin: { l: 0, t: 0, b: 0, r: 0 },showlegend: true,};\nPlotly.react("legend", data, newLayout, { displayModeBar: false, responsive: true });')

# print("Done!")


# '''
#    ╔════════════════════════════════════════════════════════════════════════╗
#    ║              HELDOUT RATES VS SMTH SPIKES (ALL CHANNELS)               ║
#    ╚════════════════════════════════════════════════════════════════════════╝
# '''
# print('Generating "HELDOUT RATES VS SMTH SPIKES (ALL CHANNELS)" plot...')

# def rates_string(neuron):
#     array_string = 'y: ['
#     for i in train_rates[:300,neuron]:
#         array_string += str(i.item())+','
#     array_string += '],'
#     return array_string

# def ss_string(neuron):
#     array_string = 'y: ['
#     for i in smth_spikes[:300,neuron]:
#         array_string += str(i.item())+','
#     array_string += '],'
#     return array_string

# with open(f"plots/{name}/spk_vs_rates_heldout_all.html", "w") as f:
#     f.write('<!DOCTYPE html><html lang="en" ><head><meta charset="UTF-8"><title>NDT Heldin Rates</title></head><body><!-- partial:index.partial.html --><div id="legend" style="height: 50px"></div><div style="height:450px; overflow-y: auto"><div id="plot" style="height:2500px"></div></div><div id="xaxis" style="height: 60px"></div><!-- partial --><script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.3.1/plotly.min.js"></script><script  src="./spk_vs_rates_heldout_all.js"></script></body></html>')

# with open(f"plots/{name}/spk_vs_rates_heldout_all.js", "w") as f:
#     names = []
#     for i in range(98, 130):
#         names.append(f'trace{i+1}')
#         names.append(f'trace{i+1}r')
#         str_to_write = f'var trace{i+1} = {{'
#         str_to_write += ss_string(i)
#         str_to_write += f"marker: {{color: '#4e79a7'}},name: 'Smoothed Spikes',yaxis: 'y{i-97}',type: 'line',"
#         if i != 0:
#             str_to_write += "showlegend: false,"
#         str_to_write += f'}};\nvar trace{i+1}r = {{'
#         str_to_write += rates_string(i)
#         str_to_write += f"marker: {{color: '#e15759'}},name: 'NDT Rates',yaxis: 'y{i-97}',type: 'line',"
#         if i != 0:
#             str_to_write += "showlegend: false,"
#         str_to_write +='};\n'
#         f.write(str_to_write)
#     names_str = 'data = ['
#     for i in names:
#         names_str += f"{i}, "
#     names_str += ']'
#     f.write(names_str+f'\n')
#     f.write(f'let bottomTraces = [{{ mode: "scatter" }}];\nlet bottomLayout = {{yaxis: {{ tickmode: "array", tickvals: [], fixedrange: true }},xaxis: {{tickmode: "array",tickvals: [0, 33, 66, 100],ticktext: ["0s", "1s", "2s", "3s"],range: [0, 100],domain: [0.0, 1.0],fixedrange: true}},margin: {{ l: 25, t: 0 , r: 40}},}};\nvar config = {{responsive: true}};\nPlotly.react("plot",data,{{xaxis: {{visible: false, fixedrange: true}},grid: {{rows: 32, columns: 1}},')
#     axis_labels = f"\nyaxis: {{title: {{text: 'ch 99',}}, showticklabels: false, fixedrange: true}},\n"
#     for i in range(100,131):
#         axis_labels += f"yaxis{i-98}: {{title: {{text: 'ch {i}',}}, showticklabels: false, fixedrange: true}},\n"
#     f.write(axis_labels)
#     f.write('margin: { l: 25, t: 25, b: 0 , r: 25},showlegend: false,},config);\nPlotly.react("xaxis", bottomTraces, bottomLayout, { displayModeBar: false, responsive: true });\ndata = [{y: [null],name: "Smooth Spikes",mode: "lines",marker: {color: "#4e79a7"},},{y: [null],name: "NDT Rates",mode: "lines",marker: {color: "#e15759"},}];\nlet newLayout = {yaxis: { visible: false},xaxis: { visible: false},margin: { l: 0, t: 0, b: 0, r: 0 },showlegend: true,};\nPlotly.react("legend", data, newLayout, { displayModeBar: false, responsive: true });')

# print("Done!")


'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                      RATES VELOCITY DECODING R^2                       ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
print('Generating "RATES VELOCITY DECODING R^2"...')

def chop_and_infer(func,
                   data,
                   seq_len=30,
                   stride=1,
                   batch_size=64,
                   output_dim=None,
                   func_kw={}
):
    device = torch.device('cuda:0')
    data_len, data_dim = data.shape[0], data.shape[1]
    output_dim = data_dim if output_dim is None else output_dim

    batch = np.zeros((batch_size, seq_len, data_dim), dtype=np.float64)
    output = np.zeros((data_len, output_dim), dtype=np.float64)
    olap = seq_len - stride

    n_seqs = (data_len - seq_len) // stride + 1
    n_batches = np.ceil(n_seqs / batch_size).astype(int)

    i_seq = 0  # index of the current sequence
    for i_batch in tqdm(range(n_batches)):
        n_seqs_batch = 0  # number of sequences in this batch
        start_ind_batch = i_seq * stride
        for i_seq_in_batch in range(batch_size):
            if i_seq < n_seqs:
                start_ind = i_seq * stride
                batch[i_seq_in_batch, :, :] = data[start_ind:start_ind +
                                                   seq_len]
                i_seq += 1
                n_seqs_batch += 1
        end_ind_batch = start_ind + seq_len
        batch_out = func(torch.Tensor(batch).to(device), **func_kw)[:n_seqs_batch]
        n_samples = n_seqs_batch * stride
        if start_ind_batch == 0:  # fill in the start of the sequence
            output[:olap, :] = batch_out[0, :olap, :].detach().cpu().numpy()
        out_idx_start = start_ind_batch + olap
        out_idx_end = end_ind_batch
        out_slice = np.s_[out_idx_start:out_idx_end]
        output[out_slice, :] = batch_out[:, olap:, :].reshape(
            n_samples, output_dim).detach().cpu().numpy()

    return output

with torch.no_grad():
    spikes = torch.Tensor(spikes_hi)
    heldout_spikes = torch.Tensor(spikes_ho)
    ho_spikes = torch.zeros_like(heldout_spikes)
    spikes_new = torch.cat([spikes, ho_spikes], -1)
    output = chop_and_infer(
        model, 
        spikes_new.numpy(),
        seq_len=30,
        stride=1
    )
rates = np.exp(output)

gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
gscv.fit(rates, vel)
print(f'NDT Decoding R2 on filtered trials: {gscv.best_score_}')
pred_vel = gscv.predict(rates)

pred_vel_df = pd.DataFrame(pred_vel, index=vel_index, columns=pd.MultiIndex.from_tuples([('pred_vel', 'x'), ('pred_vel', 'y')]))
dataset.data = pd.concat([dataset.data, pred_vel_df], axis=1)

for i in range(rates.shape[1]):
    mean = rates[:,i].mean()
    std = rates[:,i].std()
    rates[:,i] -= mean
    rates[:,i] /= std

pca = PCA(n_components=3)
pca.fit(rates)
pca_comps = pca.transform(rates)

pca_df = pd.DataFrame(pca_comps, index=vel_index, columns=pd.MultiIndex.from_tuples([('pca', 'x'), ('pca', 'y'), ('pca', 'z')]))

print("Done!")


'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                    SMTH SPIKES VELOCITY DECODING R^2                   ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
print('Generating "SMTH SPIKES VELOCITY DECODING R^2"...')

rates = dataset.data.spikes_smth_50[~nans.to_numpy() & ~nans.shift(-lag_bins, fill_value=True).to_numpy()].to_numpy()

gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
gscv.fit(rates, vel)
print(f'Smoothed Spikes Decoding R2 on filtered trials: {gscv.best_score_}')
smth_pred_vel = gscv.predict(rates)

smth_pred_vel_df = pd.DataFrame(smth_pred_vel, index=vel_index, columns=pd.MultiIndex.from_tuples([('smth_pred_vel', 'x'), ('smth_pred_vel', 'y')]))
dataset.data = pd.concat([dataset.data, smth_pred_vel_df], axis=1)

for i in range(rates.shape[1]):
    mean = rates[:,i].mean()
    std = rates[:,i].std()
    rates[:,i] -= mean
    rates[:,i] /= std

pca = PCA(n_components=3)
pca.fit(rates)
pca_comps = pca.transform(rates)

pca_df = pd.DataFrame(pca_comps, index=vel_index, columns=pd.MultiIndex.from_tuples([('smth_pca', 'x'), ('smth_pca', 'y'), ('smth_pca', 'z')]))
dataset.data = pd.concat([dataset.data, pca_df], axis=1)

print("Done!")


'''
   ╔════════════════════════════════════════════════════════════════════════╗
   ║                 RATES PREDICTED VELOCITY (TRIAL SLIDER)                ║
   ╚════════════════════════════════════════════════════════════════════════╝
'''
print('Generating "RATES PREDICTED VELOCITY (TRIAL SLIDER)" plot...')

trial_data = dataset.make_trial_data(align_field='speed_onset', align_range=(-290, 750), allow_nans=True)

fig = go.Figure()

for tid, trial in trial_data.groupby('trial_id'):
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#e15759"),
            x=np.cumsum(trial.pred_vel.to_numpy()[29:, 0]), 
            y=np.cumsum(trial.pred_vel.to_numpy()[29:, 1]), 
            name="NDT Predicted Velocity",
        ),
    )
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#4e79a7"),
            x=np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 0]), 
            y=np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 1]), 
            name="Smoothed Spikes Predicted Velocity",
        ),
    )
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#000000"),
            x=np.cumsum(trial.finger_vel.to_numpy()[29:, 0]), 
            y=np.cumsum(trial.finger_vel.to_numpy()[29:, 1]), 
            name="True Velocity",
        ),
    )

ranges = []
for tid, trial in trial_data.groupby('trial_id'):
    min_x = min(
        min(np.cumsum(trial.pred_vel.to_numpy()[29:, 0])), 
        min(np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 0])), 
        min(np.cumsum(trial.finger_vel.to_numpy()[29:, 0]))
    )
    min_y = min(
        min(np.cumsum(trial.pred_vel.to_numpy()[29:, 1])), 
        min(np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 1])), 
        min(np.cumsum(trial.finger_vel.to_numpy()[29:, 1]))
    )
    max_x = max(
        max(np.cumsum(trial.pred_vel.to_numpy()[29:, 0])), 
        max(np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 0])), 
        max(np.cumsum(trial.finger_vel.to_numpy()[29:, 0]))
    )
    max_y = max(
        max(np.cumsum(trial.pred_vel.to_numpy()[29:, 1])), 
        max(np.cumsum(trial.smth_pred_vel.to_numpy()[29:, 1])), 
        max(np.cumsum(trial.finger_vel.to_numpy()[29:, 1]))
    )
    pad = 0.05
    x_len = (max_x - min_x) * pad
    y_len = (max_y - min_y) * pad
    ranges.append(([
        min_x - x_len, 
        max_x + x_len
    ], [
        min_y - y_len, 
        max_y + y_len
    ]))
    # ranges.append(([
    #     min_x - x_len if min_x <= 0 else min_x + x_len, 
    #     max_x + x_len if max_x >= 0 else max_x - x_len
    # ], [
    #     min_y - y_len if min_y <= 0 else min_y + y_len, 
    #     max_y + y_len if max_y >= 0 else max_y - y_len
    # ]))

fig.data[0].visible = True
fig.data[1].visible = True
fig.data[2].visible = True

steps = []
for i in range(int(len(fig.data)/3)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
            #   {"title": "Slider switched to step: " + str(i), 
              {"xaxis" : dict(
                range=ranges[i][0], 
                tickmode = 'linear',
                tick0=0,
                dtick=1000, 
                zeroline=True, 
                zerolinewidth=2, 
                zerolinecolor='slategray'
              ),
              "yaxis" : dict(
                scaleanchor = "x", 
                scaleratio = 1, 
                range=ranges[i][1], 
                zeroline=True, 
                zerolinewidth=2, 
                zerolinecolor='slategray',
                tickmode = 'linear',
                tick0=0,
                dtick=1000, 
              )}],
        label=f'{i}'
    )
    step["args"][0]["visible"][i*3] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i*3+1] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i*3+2] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Trial: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders,
    legend=dict(
        yanchor="bottom",
        y=1.035,
        xanchor="right",
        x=1.00
    )
)

fig.update_xaxes(
    range=ranges[0][0], 
    tickmode = 'linear',
    tick0=0,
    dtick=1000, 
    zeroline=True, 
    zerolinewidth=2, 
    zerolinecolor='slategray'
)
fig.update_yaxes(
    scaleanchor = "x", 
    scaleratio = 1, 
    range=ranges[0][1], 
    zeroline=True, 
    zerolinewidth=2, 
    zerolinecolor='slategray',
    tickmode = 'linear',
    tick0=0,
    dtick=1000, 
)
layout = go.Layout(
    margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=25, #bottom margin
        t=25  #top margin
    )
)
fig.update_layout(layout)

fig.write_html(f"plots/{name}/velocity_slider.html")
test = 0.0 / 0.0

# trials = trial_data.trial_id.unique()

# for tid, trial in trial_data.groupby('trial_id'):
#     if tid in trials[:5]:
#         fig = plt.figure(figsize=(8,8))
#         angle = dataset.trial_info[dataset.trial_info.trial_id == tid].reach_angle
#         plt.plot(np.cumsum(trial.pred_vel.to_numpy()[29:, 0]), np.cumsum(trial.pred_vel.to_numpy()[29:, 1]), color='tab:red', label='NDT Predicted Velocity')
#         plt.plot(
#             np.cumsum(trial.smth_pred_vel.x.to_numpy()[29:]), 
#             np.cumsum(trial.smth_pred_vel.y.to_numpy()[29:]), 
#             color='tab:blue', 
#             label='Smoothing Predicted Velocity'
#         )
#         plt.plot(
#             np.cumsum(trial.finger_vel.x.to_numpy()[29:]), 
#             np.cumsum(trial.finger_vel.y.to_numpy()[29:]), 
#             color='black', 
#             label='True Velocity'
#         )
#         plt.legend()
#         plt.savefig(f'test_vel_{tid}.png')

print("Done!")

"""
SMTH SPIKES PREDICTED VELOCITY (TRIAL SLIDER)
"""


"""
RATES PCA PLOT
"""


"""
SMTH SPIKES PCA PLOT
"""



# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import math
# fig = plt.figure(figsize=(5,5))
# xval = np.arange(-math.pi, math.pi, 0.01)
# yval = np.ones_like(xval)

# colormap = plt.get_cmap('hsv')
# norm = mpl.colors.Normalize(-math.pi, np.pi)

# ax = plt.subplot(1, 1, 1, polar=True)
# ax.scatter(xval, yval, c=xval, s=300, cmap=colormap, norm=norm, linewidths=5)
# ax.set_rmin(0.9)
# ax.set_rmax(1.01)
# ax.set_yticks([])
# plt.savefig('ring.png')

def chop_and_infer(func,
                   data,
                   seq_len=30,
                   stride=1,
                   batch_size=64,
                   output_dim=None,
                   func_kw={}
):
    device = torch.device('cuda:0')
    data_len, data_dim = data.shape[0], data.shape[1]
    output_dim = data_dim if output_dim is None else output_dim

    batch = np.zeros((batch_size, seq_len, data_dim), dtype=np.float64)
    output = np.zeros((data_len, output_dim), dtype=np.float64)
    olap = seq_len - stride

    n_seqs = (data_len - seq_len) // stride + 1
    n_batches = np.ceil(n_seqs / batch_size).astype(int)

    i_seq = 0  # index of the current sequence
    for i_batch in tqdm(range(n_batches)):
        n_seqs_batch = 0  # number of sequences in this batch
        start_ind_batch = i_seq * stride
        for i_seq_in_batch in range(batch_size):
            if i_seq < n_seqs:
                start_ind = i_seq * stride
                batch[i_seq_in_batch, :, :] = data[start_ind:start_ind +
                                                   seq_len]
                i_seq += 1
                n_seqs_batch += 1
        end_ind_batch = start_ind + seq_len
        batch_out = func(torch.Tensor(batch).to(device), **func_kw)[:n_seqs_batch]
        n_samples = n_seqs_batch * stride
        if start_ind_batch == 0:  # fill in the start of the sequence
            output[:olap, :] = batch_out[0, :olap, :].detach().cpu().numpy()
        out_idx_start = start_ind_batch + olap
        out_idx_end = end_ind_batch
        out_slice = np.s_[out_idx_start:out_idx_end]
        output[out_slice, :] = batch_out[:, olap:, :].reshape(
            n_samples, output_dim).detach().cpu().numpy()

    return output


if len(sys.argv) == 1 or len(sys.argv) > 2:
        print("Invalid Arguments...\n\nYou must supply a path to a '.pt' file.")
        exit()
path = sys.argv[1]
name = path[:path.rindex('/')].split('/')[-1]
config = get_config_from_file(path[:path.rindex('/')+1]+'config.yaml')

set_device(config)
device = torch.device('cuda:0')

set_seeds(config)

model = torch.load(path)
# model = torch.load(path).to(device)
model.name = name
model.eval()

make_test_data()
with h5py.File('/home/dmifsud/Projects/NDT-Timing-Test/data/mc_rtt_cont_24_test.h5', 'r') as h5file:
    h5dict = h5_to_dict(h5file)


dataset = NWBDataset('/home/dmifsud/Projects/NDT-U/data/mc_rtt_train.nwb', split_heldout=True)

# Find when target pos changes
has_change = dataset.data.target_pos.fillna(-1000).diff(axis=0).any(axis=1) # filling NaNs with arbitrary scalar to treat as one block
# Find if target pos change corresponds to NaN-padded gap between files
change_nan = dataset.data[has_change].isna().any(axis=1)
# Drop trials containing the gap and immediately before and after, as those trials may be cut short
drop_trial = (change_nan | change_nan.shift(1, fill_value=True) | change_nan.shift(-1, fill_value=True))[:-1]
# Add start and end times to trial info
change_times = dataset.data.index[has_change]
start_times = change_times[:-1][~drop_trial]
end_times = change_times[1:][~drop_trial]
# Get target position per trial
target_pos = dataset.data.target_pos.loc[start_times].to_numpy().tolist()
# Compute reach distance and angle
reach_dist = dataset.data.target_pos.loc[end_times - pd.Timedelta(1, 'ms')].to_numpy() - dataset.data.target_pos.loc[start_times - pd.Timedelta(1, 'ms')].to_numpy()
reach_angle = np.arctan2(reach_dist[:, 1], reach_dist[:, 0]) / np.pi * 180
# Create trial info
dataset.trial_info = pd.DataFrame({
    'trial_id': np.arange(len(start_times)),
    'start_time': start_times,
    'end_time': end_times,
    'target_pos': target_pos,
    'reach_dist_x': reach_dist[:, 0],
    'reach_dist_y': reach_dist[:, 1],
    'reach_angle': reach_angle,
})

dataset.resample(10)

speed = np.linalg.norm(dataset.data.finger_vel, axis=1)
dataset.data['speed'] = speed
peak_times = dataset.calculate_onset('speed', 0.05)

dataset.smooth_spk(50, name='smth_50', ignore_nans=True)

# Lag velocity by 120 ms relative to neural data
lag = 120
lag_bins = int(round(lag / dataset.bin_width))
nans = dataset.data.finger_vel.x.isna().reset_index(drop=True)

spikes_hi = dataset.data.spikes[~nans.to_numpy() & ~nans.shift(-lag_bins, fill_value=True).to_numpy()].to_numpy()
spikes_ho = dataset.data.heldout_spikes[~nans.to_numpy() & ~nans.shift(-lag_bins, fill_value=True).to_numpy()].to_numpy()

vel = dataset.data.finger_vel[~nans.to_numpy() & ~nans.shift(lag_bins, fill_value=True).to_numpy()].to_numpy()
vel_index = dataset.data.finger_vel[~nans.to_numpy() & ~nans.shift(lag_bins, fill_value=True).to_numpy()].index

with torch.no_grad():
    train_rates = []
    # for step, (spikes, heldout_spikes) in enumerate(test_dataloader):
    # for spikes, heldout_spikes in zip(
    #     torch.Tensor(spikes_hi).to(device), torch.Tensor(spikes_ho).to(device)
    # ):
    spikes = torch.Tensor(spikes_hi)
    heldout_spikes = torch.Tensor(spikes_ho)
    ho_spikes = torch.zeros_like(heldout_spikes)
    spikes_new = torch.cat([spikes, ho_spikes], -1)
    print(spikes_new.shape)
    output = chop_and_infer(
        model, 
        spikes_new.numpy(),
        seq_len=30,
        stride=1,
        batch_size=64,
        output_dim=None,
        func_kw={}
    )
rates = np.exp(output)


gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
gscv.fit(rates, vel)
print(f'NDT Decoding R2: {gscv.best_score_}')
pred_vel = gscv.predict(rates)

# Add data back to main dataframe
pred_vel_df = pd.DataFrame(pred_vel, index=vel_index, columns=pd.MultiIndex.from_tuples([('pred_vel', 'x'), ('pred_vel', 'y')]))
dataset.data = pd.concat([dataset.data, pred_vel_df], axis=1)

for i in range(rates.shape[1]):
    mean = rates[:,i].mean()
    std = rates[:,i].std()
    rates[:,i] -= mean
    rates[:,i] /= std

pca = PCA(n_components=3)
pca.fit(rates)
pca_comps = pca.transform(rates)

pca_df = pd.DataFrame(pca_comps, index=vel_index, columns=pd.MultiIndex.from_tuples([('pca', 'x'), ('pca', 'y'), ('pca', 'z')]))
dataset.data = pd.concat([dataset.data, pca_df], axis=1)

rates = dataset.data.spikes_smth_50[~nans.to_numpy() & ~nans.shift(-lag_bins, fill_value=True).to_numpy()].to_numpy()

gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
gscv.fit(rates, vel)
print(f'Smoothing Decoding R2: {gscv.best_score_}')
smth_pred_vel = gscv.predict(rates)

# Add data back to main dataframe
smth_pred_vel_df = pd.DataFrame(smth_pred_vel, index=vel_index, columns=pd.MultiIndex.from_tuples([('smth_pred_vel', 'x'), ('smth_pred_vel', 'y')]))
dataset.data = pd.concat([dataset.data, smth_pred_vel_df], axis=1)

for i in range(rates.shape[1]):
    mean = rates[:,i].mean()
    std = rates[:,i].std()
    rates[:,i] -= mean
    rates[:,i] /= std

pca = PCA(n_components=3)
pca.fit(rates)
pca_comps = pca.transform(rates)

pca_df = pd.DataFrame(pca_comps, index=vel_index, columns=pd.MultiIndex.from_tuples([('smth_pca', 'x'), ('smth_pca', 'y'), ('smth_pca', 'z')]))
dataset.data = pd.concat([dataset.data, pca_df], axis=1)

trial_data = dataset.make_trial_data(align_field='speed_onset', align_range=(-290, 750), allow_nans=True)

max_angle = 180
min_angle = -180
# max_angle = -1000
# min_angle = 1000
# for tid, trial in trial_data.groupby('trial_id'):
#     angle = dataset.trial_info[dataset.trial_info.trial_id == tid].reach_angle
#     if angle.item() < min_angle:
#         min_angle = angle.item()
#     if angle.item() > max_angle:
#         max_angle = angle.item()

norm = colors.Normalize(vmin=min_angle, vmax=max_angle, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap='hsv')
mapper.set_array([])

# ax = plt.axes(projection='3d')
size = 5000
# plt.xlim([-size,size])
# plt.ylim([-size,size])
# plt.xlim([-9000,9000])
# plt.ylim([-9000,9000])
# plt.xlim([-500,500])
# plt.ylim([-500,500])
trials = trial_data.trial_id.unique()

for tid, trial in trial_data.groupby('trial_id'):
    if tid in trials[:5]:
        fig = plt.figure(figsize=(8,8))
        angle = dataset.trial_info[dataset.trial_info.trial_id == tid].reach_angle
        plt.plot(np.cumsum(trial.pred_vel.to_numpy()[29:, 0]), np.cumsum(trial.pred_vel.to_numpy()[29:, 1]), color='tab:red', label='NDT Predicted Velocity')
        plt.plot(
            np.cumsum(trial.smth_pred_vel.x.to_numpy()[29:]), 
            np.cumsum(trial.smth_pred_vel.y.to_numpy()[29:]), 
            color='tab:blue', 
            label='Smoothing Predicted Velocity'
        )
        plt.plot(
            np.cumsum(trial.finger_vel.x.to_numpy()[29:]), 
            np.cumsum(trial.finger_vel.y.to_numpy()[29:]), 
            color='black', 
            label='True Velocity'
        )
        plt.legend()
        plt.savefig(f'test_vel_{tid}.png')
        # plt.plot(trial.finger_vel.x, trial.finger_vel.y, color=mapper.to_rgba(angle), alpha=0.5)
        # ax.plot3D(trial.pca.z, trial.pca.y, trial.pca.x, color=mapper.to_rgba(angle), alpha=0.5)

fig = plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')
# plt.xlim([-500,500])
# plt.ylim([-500,500])

for tid, trial in trial_data.groupby('trial_id'):
    angle = dataset.trial_info[dataset.trial_info.trial_id == tid].reach_angle
    # plt.plot(trial.finger_vel.x, trial.finger_vel.y, color=mapper.to_rgba(angle), alpha=0.5)
    ax.plot3D(trial.pca.z, trial.pca.y, trial.pca.x, color=mapper.to_rgba(angle), alpha=0.5)
    # plt.plot(np.cumsum(trial.finger_vel.x), np.cumsum(trial.finger_vel.y), color=mapper.to_rgba(angle))
    # plt.plot(np.cumsum(trial.finger_vel.x[29:]), np.cumsum(trial.finger_vel.y[29:]), color=mapper.to_rgba(angle))
        # plt.plot(trial.cursor_pos.x*50, trial.cursor_pos.y*50, color='blue')
plt.savefig('test_pca_z_y_x.png')
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
for tid, trial in trial_data.groupby('trial_id'):
    angle = dataset.trial_info[dataset.trial_info.trial_id == tid].reach_angle
    # plt.plot(trial.finger_vel.x, trial.finger_vel.y, color=mapper.to_rgba(angle), alpha=0.5)
    ax.plot3D(trial.pca.x, trial.pca.y, trial.pca.z, color=mapper.to_rgba(angle), alpha=0.5)
    # plt.plot(np.cumsum(trial.finger_vel.x), np.cumsum(trial.finger_vel.y), color=mapper.to_rgba(angle))
    # plt.plot(np.cumsum(trial.finger_vel.x[29:]), np.cumsum(trial.finger_vel.y[29:]), color=mapper.to_rgba(angle))
        # plt.plot(trial.cursor_pos.x*50, trial.cursor_pos.y*50, color='blue')
cbar = plt.colorbar(mapper, ticks=[-180, -90, 0, 90, 180])
cbar.ax.set_yticklabels(['-180°', '-90°', '0°', '90°', '180°'])
plt.savefig('test_pca_x_y_z.png')
fig = plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')
for tid, trial in trial_data.groupby('trial_id'):
    angle = dataset.trial_info[dataset.trial_info.trial_id == tid].reach_angle
    # plt.plot(trial.finger_vel.x, trial.finger_vel.y, color=mapper.to_rgba(angle), alpha=0.5)
    ax.plot3D(trial.pca.y, trial.pca.z, trial.pca.x, color=mapper.to_rgba(angle), alpha=0.5)
    # plt.plot(np.cumsum(trial.finger_vel.x), np.cumsum(trial.finger_vel.y), color=mapper.to_rgba(angle))
    # plt.plot(np.cumsum(trial.finger_vel.x[29:]), np.cumsum(trial.finger_vel.y[29:]), color=mapper.to_rgba(angle))
        # plt.plot(trial.cursor_pos.x*50, trial.cursor_pos.y*50, color='blue')
plt.savefig('test_pca_y_z_x.png')
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
for tid, trial in trial_data.groupby('trial_id'):
    angle = dataset.trial_info[dataset.trial_info.trial_id == tid].reach_angle
    # plt.plot(trial.finger_vel.x, trial.finger_vel.y, color=mapper.to_rgba(angle), alpha=0.5)
    ax.plot3D(trial.pca.y, trial.pca.x, trial.pca.z, color=mapper.to_rgba(angle), alpha=0.5)
    # plt.plot(np.cumsum(trial.finger_vel.x), np.cumsum(trial.finger_vel.y), color=mapper.to_rgba(angle))
    # plt.plot(np.cumsum(trial.finger_vel.x[29:]), np.cumsum(trial.finger_vel.y[29:]), color=mapper.to_rgba(angle))
        # plt.plot(trial.cursor_pos.x*50, trial.cursor_pos.y*50, color='blue')
cbar = plt.colorbar(mapper, ticks=[-180, -90, 0, 90, 180])
cbar.ax.set_yticklabels(['-180°', '-90°', '0°', '90°', '180°'])
plt.savefig('test_pca_y_x_z.png')


fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
for tid, trial in trial_data.groupby('trial_id'):
    angle = dataset.trial_info[dataset.trial_info.trial_id == tid].reach_angle
    # plt.plot(trial.finger_vel.x, trial.finger_vel.y, color=mapper.to_rgba(angle), alpha=0.5)
    ax.plot3D(trial.smth_pca.y, trial.smth_pca.x, trial.smth_pca.z, color=mapper.to_rgba(angle), alpha=0.5)
    # plt.plot(np.cumsum(trial.finger_vel.x), np.cumsum(trial.finger_vel.y), color=mapper.to_rgba(angle))
    # plt.plot(np.cumsum(trial.finger_vel.x[29:]), np.cumsum(trial.finger_vel.y[29:]), color=mapper.to_rgba(angle))
        # plt.plot(trial.cursor_pos.x*50, trial.cursor_pos.y*50, color='blue')
cbar = plt.colorbar(mapper, ticks=[-180, -90, 0, 90, 180])
cbar.ax.set_yticklabels(['-180°', '-90°', '0°', '90°', '180°'])
plt.savefig('test_smth_pca_y_x_z.png')


fig = plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')
for tid, trial in trial_data.groupby('trial_id'):
    angle = dataset.trial_info[dataset.trial_info.trial_id == tid].reach_angle
    # plt.plot(trial.finger_vel.x, trial.finger_vel.y, color=mapper.to_rgba(angle), alpha=0.5)
    ax.plot3D(trial.pca.z, trial.pca.x, trial.pca.y, color=mapper.to_rgba(angle), alpha=0.5)
    # plt.plot(np.cumsum(trial.finger_vel.x), np.cumsum(trial.finger_vel.y), color=mapper.to_rgba(angle))
    # plt.plot(np.cumsum(trial.finger_vel.x[29:]), np.cumsum(trial.finger_vel.y[29:]), color=mapper.to_rgba(angle))
        # plt.plot(trial.cursor_pos.x*50, trial.cursor_pos.y*50, color='blue')
plt.savefig('test_pca_z_x_y.png')
# with torch.no_grad():
#     trial_rates = []
#     for tid, trial in trial_data.groupby('trial_id'):
#         hi_spikes = torch.Tensor(trial.spikes.to_numpy())
#         heldout_spikes = torch.Tensor(trial.heldout_spikes.to_numpy())
#         # hi_spikes = torch.Tensor(np.expand_dims(trial.spikes, 0)).to(device)
#         # heldout_spikes = torch.Tensor(np.expand_dims(trial.heldout_spikes, 0)).to(device)
#         ho_spikes = torch.zeros_like(heldout_spikes)
#         # ho_spikes = torch.zeros_like(heldout_spikes, device=device)
#         spikes_new = torch.cat([hi_spikes, heldout_spikes], -1)
#         # print(spikes_new.shape)
#         output = chop_and_infer(
#             model, 
#             spikes_new,
#             seq_len=30,
#             stride=1,
#             batch_size=64,
#             output_dim=None,
#             func_kw={}
#         )
#         trial_rates.append(output)

# trial_rates_tensor = np.exp(np.concatenate(trial_rates, 0))
# print(vel.shape)
# print(trial_rates_tensor.shape)




# with torch.no_grad():
#     train_rates = []
#     spikes_test = []
#     # for step, (spikes, heldout_spikes) in enumerate(test_dataloader):
#     for spikes, heldout_spikes in zip(
#         torch.Tensor(spikes_hi).to(device), torch.Tensor(spikes_ho).to(device)
#     ):
#         ho_spikes = torch.zeros_like(heldout_spikes, device=device)
#         spikes_new = torch.cat([spikes, ho_spikes], -1)
#         output = model(spikes_new.unsqueeze(dim=0))[:, -1, :]
#         train_rates.append(output.cpu())
#         # spikes_full = torch.cat([spikes, heldout_spikes], -1)
#         # spikes_test.append(spikes_full[:, -1, :].cpu())

#     train_rates = torch.cat(train_rates, dim=0).exp() # turn into tensor and use exponential on rates
#     # spikes_test = torch.cat(spikes_test, dim=0).numpy() # turn into tensor and use exponential on rates

# # Add data back to main dataframe
# pred_vel_df = pd.DataFrame(pred_vel, index=vel_index, columns=pd.MultiIndex.from_tuples([('pred_vel', 'x'), ('pred_vel', 'y')]))
# dataset.data = pd.concat([dataset.data, pred_vel_df], axis=1)
# # Add data back to main dataframe
# pred_vel_df = pd.DataFrame(test2, index=vel_index, columns=pd.MultiIndex.from_tuples([('pca', 'x'), ('pca', 'y'),('pca', 'z'),]))
# dataset.data = pd.concat([dataset.data, pred_vel_df], axis=1)

# # Lag velocity by 120 ms relative to neural data
# lag = 120
# lag_bins = int(round(lag / dataset.bin_width))
# nans = dataset.data.finger_vel.x.isna().reset_index(drop=True)
# rates = dataset.data.spikes_smth_50[~nans.to_numpy() & ~nans.shift(-lag_bins, fill_value=True).to_numpy()].to_numpy()
# vel = dataset.data.finger_vel[~nans.to_numpy() & ~nans.shift(lag_bins, fill_value=True).to_numpy()].to_numpy()
# vel_index = dataset.data.finger_vel[~nans.to_numpy() & ~nans.shift(lag_bins, fill_value=True).to_numpy()].index

# from sklearn.decomposition import PCA
# pca = PCA(n_components=3)
# pca.fit(rates)
# test2 = pca.transform(rates)

# # Fit decoder and evaluate
# gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
# gscv.fit(rates, vel)
# print(f'Decoding R2: {gscv.best_score_}')
# pred_vel = gscv.predict(rates)

# # Add data back to main dataframe
# pred_vel_df = pd.DataFrame(pred_vel, index=vel_index, columns=pd.MultiIndex.from_tuples([('pred_vel', 'x'), ('pred_vel', 'y')]))
# dataset.data = pd.concat([dataset.data, pred_vel_df], axis=1)
# # Add data back to main dataframe
# pred_vel_df = pd.DataFrame(test2, index=vel_index, columns=pd.MultiIndex.from_tuples([('pca', 'x'), ('pca', 'y'),('pca', 'z'),]))
# dataset.data = pd.concat([dataset.data, pred_vel_df], axis=1)

# trial_data = dataset.make_trial_data(align_field='speed_onset', align_range=(0, 750), allow_nans=True)


# max_angle = -1000
# min_angle = 1000
# for tid, trial in trial_data.groupby('trial_id'):
#     angle = dataset.trial_info[dataset.trial_info.trial_id == tid].reach_angle
#     if angle.item() < min_angle:
#         min_angle = angle.item()
#     if angle.item() > max_angle:
#         max_angle = angle.item()

# norm = colors.Normalize(vmin=min_angle, vmax=max_angle, clip=True)
# mapper = cm.ScalarMappable(norm=norm, cmap='hsv')

# fig = plt.figure(figsize=(15,15))
# # ax = plt.axes(projection='3d')
# plt.xlim([-9000,9000])
# plt.ylim([-9000,9000])
# # plt.xlim([-500,500])
# # plt.ylim([-500,500])

# for tid, trial in trial_data.groupby('trial_id'):
#     angle = dataset.trial_info[dataset.trial_info.trial_id == tid].reach_angle
#     # plt.plot(trial.finger_vel.x, trial.finger_vel.y, color=mapper.to_rgba(angle), alpha=0.5)
#     # ax.plot3D(trial.pca.z, trial.pca.y, trial.pca.x, color=mapper.to_rgba(angle), alpha=0.5)
#     plt.plot(np.cumsum(trial.finger_vel.x), np.cumsum(trial.finger_vel.y), color=mapper.to_rgba(angle))
#     # plt.plot(np.cumsum(trial.finger_vel.x[29:]), np.cumsum(trial.finger_vel.y[29:]), color=mapper.to_rgba(angle))
#         # plt.plot(trial.cursor_pos.x*50, trial.cursor_pos.y*50, color='blue')


with torch.no_grad():
    train_rates = []
    for spikes, heldout_spikes in zip(
        torch.Tensor(h5dict['test_spikes_heldin']).to(device), torch.Tensor(h5dict['test_spikes_heldout']).to(device)
    ):
        ho_spikes = torch.zeros_like(heldout_spikes, device=device)
        spikes_new = torch.cat([spikes, ho_spikes], -1)
        output = model(spikes_new.unsqueeze(dim=0))[:, -1, :]
        train_rates.append(output.cpu())

    train_rates = torch.cat(train_rates, dim=0).exp() # turn into tensor and use exponential on rates

spikes = torch.Tensor(h5dict['test_hi_smth_spikes'])
heldout_spikes = torch.Tensor(h5dict['test_ho_smth_spikes'])
smth_spikes = torch.cat([spikes, heldout_spikes], -1)

# print(train_rates.numpy().shape)
# print(h5dict['test_vel_segments'].shape)

# rates = train_rates.numpy()
# vel = h5dict['test_vel_segments']

# # from nlb_tools.nwb_interface import NWBDataset
# # import numpy as np
# # import pandas as pd
# # import scipy.signal as signal
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# # from matplotlib.collections import LineCollection
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import GridSearchCV
# import matplotlib.pyplot as plt

# gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
# gscv.fit(rates, vel)
# print(f'Rates Decoding R2: {gscv.best_score_}')
# pred_vel = gscv.predict(rates)

# plt.plot(np.cumsum(pred_vel[0, 0]), np.cumsum(pred_vel[0, 1]))
# plt.savefig('test.png')

# rates = np.concatenate((h5dict['test_hi_smth_spikes'], h5dict['test_ho_smth_spikes']), -1)
# vel = h5dict['test_vel_segments']

# gscv = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 0, 9)})
# gscv.fit(rates, vel)
# print(f'Smooth Spikes Decoding R2: {gscv.best_score_}')
# pred_vel = gscv.predict(rates)


# Create figure
fig = go.Figure()

x_range=2500

fig.add_trace(
    go.Scatter(y=list(train_rates[:x_range,0]), line=dict(color="#e15759"), name="NDT Rates",))
fig.add_trace(
    go.Scatter(y=list(smth_spikes[:x_range,0]), line=dict(color="#4e79a7"), name="Smooth Spikes",))
for i in range(1, 98):
    fig.add_trace(
        go.Scatter(y=list(train_rates[:x_range,i]), visible=False, line=dict(color="#e15759"), name="NDT Rates",))
    fig.add_trace(
        go.Scatter(y=list(smth_spikes[:x_range,i]), visible=False, line=dict(color="#4e79a7"), name="Smooth Spikes",))

# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
        ),
        rangeslider=dict(
            visible=True
        )
    )
)

buttons = []
for i in range(98):
    vis_list = [False for i in range(196)]
    vis_list[i*2] = True
    vis_list[i*2+1] = True
    buttons.append(dict(
        method='restyle',
        label=f'ch {i+1}',
        visible=True,
        args=[{'visible':vis_list,}]
    ))
           
# specify updatemenu        
um = [{
    'buttons':buttons, 
    'direction': 'down',
    'pad': {"r": 0, "t": 0},
    'showactive':True,
    'x':0.0,
    'xanchor':"left",
    'y':1.075,
    'yanchor':"bottom" 
}]

fig.update_layout(updatemenus=um)

initial_range = [
    '0', '300'
]

fig['layout']['xaxis'].update(range=initial_range)

layout = go.Layout(
  margin=go.layout.Margin(
        l=0, #left margin
        r=0, #right margin
        b=0, #bottom margin
        t=0  #top margin
    )
)
fig.update_layout(layout)

fig.update_xaxes(
    ticktext=[f'{int(i/100)}s' for i in range(0, x_range, 100)],
    tickvals=[i for i in range(0, x_range, 100)],
)

fig.update_layout(legend=dict(
    yanchor="bottom",
    y=1.035,
    xanchor="right",
    x=1.00
))

fig.write_html("heldin_single.html")