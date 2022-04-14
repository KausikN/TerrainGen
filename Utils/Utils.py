'''
Utils for TerrainGen
'''

# Imports
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as graph
from PIL import Image

from .MaskFunctions import *
from .ColoriserFunctions import *

# Main Functions
# Depth Functions
def DepthFunc_GreyScaleDepth(I, **params):
    '''
    Get GrayScale Depth from Image
    '''
    if I.ndim == 3:
        I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    return I

# Plot Functions
def PlotImage3D_Plane(I, Depths, DepthLimits=None, fig=None, display=True):
    if DepthLimits is None:
        DepthLimits = [np.min(Depths)-0, np.max(Depths)+1]

    Z = Depths
    facecolors = np.array(cv2.cvtColor(I, cv2.COLOR_BGR2RGBA), dtype=np.uint8)

    I_8bit = Image.fromarray(facecolors).convert('P', palette='WEB', dither=None)
    I_idx = Image.fromarray(facecolors).convert('P', palette='WEB')
    idx_to_color = np.array(I_idx.getpalette()).reshape((-1, 3))
    colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]

    if fig is None:
        fig = graph.Figure(data=[graph.Surface(
            z=Z, surfacecolor=I_8bit, cmin=0, cmax=255, colorscale=colorscale, showscale=False
        )])
        fig.update_layout(
            title='',
            autosize=True,
            scene=dict(
                zaxis=dict(range=[0.0, 1.0])
            )
        )
    else:
        fig.update_traces(
            z=Z, surfacecolor=I_8bit
        )
    
    if display:
        plt.show()

    return fig

# Driver Code
# Params

# Params

# RunCode