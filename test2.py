from plotly.offline import plot
import plotly.graph_objects as go
import numpy as np
import data_V2

x = predictions.flatten()
predictions_new = x.tolist()

databas = data_V2.dataset.flatten()
databas = databas.tolist()


fig = go.Figure()
fig.add_trace(go.Scatter(x=trained_val, y=databas, name="Actaul Value",
                         line_color='deepskyblue'))

fig.add_trace(go.Scatter(x=test_val, y=predictions_new, name="Predicted Value",
                         line_color='dimgray'))

fig.update_layout(title_text='Time Series with Rangeslider',
                  xaxis_rangeslider_visible=True)
plot(fig)