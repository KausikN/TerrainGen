'''
Utils for TerrainGen
'''

# Imports
import cv2
import numpy as np

# Main Functions
def DepthFunc_GreyScaleDepth(I, options=None):
    if I.ndim == 3:
        I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    return I


# Driver Code
# Params

# Params

# RunCode