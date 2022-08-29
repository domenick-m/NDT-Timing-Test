import sys
import torch
import numpy as np
import scipy.signal as signal
import plotly.graph_objects as go
from datasets import get_dataloaders
from setup import set_device, set_seeds
from configs.default_config import get_config_from_file

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

test_dataloader = get_dataloaders(config, 'test')

with torch.no_grad():
    train_rates = []
    spikes_test = []
    # for step, (spikes, heldout_spikes) in enumerate(test_dataloader):
    for step, (spikes, heldout_spikes) in enumerate(test_dataloader):
        ho_spikes = torch.zeros_like(heldout_spikes, device=device)
        spikes_new = torch.cat([spikes, ho_spikes], -1)
        output = model(spikes_new)[:, -1, :]
        train_rates.append(output.cpu())
        spikes_full = torch.cat([spikes, heldout_spikes], -1)
        spikes_test.append(spikes_full[:, -1, :].cpu())

    train_rates = torch.cat(train_rates, dim=0).exp() # turn into tensor and use exponential on rates
    spikes_test = torch.cat(spikes_test, dim=0).numpy() # turn into tensor and use exponential on rates

smooth_std = 30 #ms
kern_sd = int(round(smooth_std / 10))
window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
window /= np.sum(window)
filt = lambda x: np.convolve(x, window, 'same')

smth_spikes = np.apply_along_axis(filt, 0, spikes_test)

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
    '0', '200'
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