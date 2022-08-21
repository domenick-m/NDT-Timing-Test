//x-axis plot info
let bottomTraces = [{ mode: "scatter" }];
let bottomLayout = {
  yaxis: { tickmode: "array", tickvals: [] },
  xaxis: {
    tickmode: "array",
    tickvals: [0, 50, 100],
    ticktext: ["0", "50", "100"],
    range: [0, 100]
  },
  margin: { l: 100, t: 0 }
};

//plot top/data plot
Plotly.react(
  "plot",
  [{ x: [1, 10, 50, 60], y: [100, -100, 50, -30], mode: "scatter" }],
  {
    xaxis: {
      range: [0, 100]
    },
    yaxis: {
      zeroline: false
    },
    margin: { l: 100, b: 0 }
  }
);

//plot x-axis
Plotly.react("xaxis", bottomTraces, bottomLayout, { displayModeBar: false , responsive: true});

//catch some top plot relayout events and redraw xaxis as an example
document.getElementById("plot").on("plotly_relayout", function (data) {
  if (data["xaxis.range[0]"]) {
    bottomLayout.xaxis.range = [data["xaxis.range[0]"], data["xaxis.range[1]"]];
  } else if (data["xaxis.autorange"]) {
    bottomLayout.xaxis.range = [0, 100];
  }
  Plotly.react("xaxis", bottomTraces, bottomLayout, { displayModeBar: false , responsive: true});
});
