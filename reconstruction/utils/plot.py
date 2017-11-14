# 3d draw function
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
from .data_prep import volume_to_point_cloud
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

def plot_vol(vol, s=10, c=(105,127,155), show_grid=False):
    if vol.dtype != np.bool:
        vol = vol > 0

    pc = volume_to_point_cloud(vol)
    plot3d(pc, s, c, show_grid)

def plot3d(verts, s=10, c=(105,127,155), show_grid=False):
    x, y, z = zip(*verts)
    color = f'rgb({c[0]}, {c[1]}, {c[2]})'
    trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=s,
            color=color,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5
            ),
            opacity=1
        )
    )
    data = [trace]
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene = go.Scene(
            xaxis=dict(visible=show_grid),
            yaxis=dict(visible=show_grid),
            zaxis=dict(visible=show_grid)
        )
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)

def plot_reconstruction(vol1, vol2, s=10,
                        c1=(105,128,155), c2=(182,49,62), show_grid=False):
    if vol1.dtype != np.bool:
        vol1 = vol1 > 0
    if vol2.dtype != np.bool:
        vol2 = vol2 > 0
        
    color1 = f'rgb({c1[0]}, {c1[1]}, {c1[2]})'
    color2 = f'rgb({c2[0]}, {c2[1]}, {c2[2]})'
    vol2 = np.logical_xor(vol2, vol1)
    pc1 = volume_to_point_cloud(vol1)
    pc2 = volume_to_point_cloud(vol2)
    x1, y1, z1 = zip(*pc1)
    x2, y2, z2 = zip(*pc2)
    trace1 = go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode='markers',
        marker=dict(
            size=s,
            color=color1,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5
            ),
            opacity=1
        )
    )
    trace2 = go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode='markers',
        marker=dict(
            size=s,
            color=color2,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5
            ),
            opacity=1
        )
    )
    data = [trace1, trace2]
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0), 
                       scene = go.Scene(
                               xaxis=dict(visible=show_grid),
                               yaxis=dict(visible=show_grid),
                               zaxis=dict(visible=show_grid)),
                       showlegend=False)
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
