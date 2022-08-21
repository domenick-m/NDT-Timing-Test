var trace1 = {
  y: [4, 5, 6],
  marker: {color: '#23b7e5'},
  name: "Smoothed Spikes",
  yaxis: 'y1',
  type: 'scatter'
};

var trace1r = {
  y: [5, 4, 4],
  marker: {color: '#ff0000'},
  name: "Rates",
  yaxis: 'y1',
  type: 'scatter'
};

var trace2 = {
  y: [0.50, 0.60, 0.70],
  marker: {color: '#23b7e5'},
  showlegend: false,
  yaxis: 'y2',
  // yaxis: 'y1',
  // xaxis: 'x2',
  type: 'scatter'
};
var trace2r = {
  y: [0.6, 0.4, 0.4],
  marker: {color: '#ff0000'},
  showlegend: false,
  yaxis: 'y2',
  // yaxis: 'y1',
  // xaxis: 'x2',
  type: 'scatter'
};

var trace3 = {
  y: [50, 60, 70],
  marker: {color: '#23b7e5'},
  showlegend: false,
  yaxis: 'y3',
  // yaxis: 'y2',
  type: 'scatter'
};
var trace3r = {
  y: [40, 40, 80],
  marker: {color: '#ff0000'},
  showlegend: false,
  yaxis: 'y3',
  // yaxis: 'y2',
  type: 'scatter'
};

var trace4 = {
  y: [50, 60, 70],
  marker: {color: '#23b7e5'},
  showlegend: false,
  yaxis: 'y4',
  // yaxis: 'y2',
  // xaxis: 'x2',
  type: 'scatter'
};

var trace5 = {
  y: [50, 60, 70],
  marker: {color: '#23b7e5'},
  showlegend: false,
  yaxis: 'y5',
  // yaxis: 'y3',
  type: 'scatter'
};

var trace6 = {
  y: [50, 60, 70],
  marker: {color: '#23b7e5'},
  showlegend: false,
  yaxis: 'y6',
  // yaxis: 'y3',
  // xaxis: 'x2',
  type: 'scatter'
};

var data = [trace1, trace1r, trace2, trace2r, trace3, trace3r, trace4, trace5, trace6];

let bottomTraces = [{ mode: "scatter" }];
let bottomLayout = {
  yaxis: { tickmode: "array", tickvals: [] },
  xaxis: {
    tickmode: "array",
    tickvals: [0, 50, 100],
    ticktext: ["0", "50", "100"],
    range: [0, 100],
    domain: [0.05, 0.915]
  },
  margin: { l: 100, t: 0 },
  
};



// var layout = {
//   grid: {rows: 98, columns: 2}, yaxis: {
//       zeroline: false
//     },
//     margin: { l: 100, b: 0 }
// };

var config = {responsive: true}

// Plotly.newPlot('myDiv', data, layout, config);
Plotly.react(
  "plot",
  data,
  {
    xaxis: {
      visible: false,
    },
    grid: {rows: 6, columns: 1},
    yaxis: {
      zeroline: true
    },
    margin: { l: 100, t: 25, b: 0 }
  }
);

Plotly.react("xaxis", bottomTraces, bottomLayout, { displayModeBar: false });
