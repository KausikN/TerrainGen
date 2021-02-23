'''
Utils for TerrainGen
'''

# Imports
import cv2
import numpy as np
from tqdm import tqdm

# Main Functions
# Depth Functions
def DepthFunc_GreyScaleDepth(I, options=None):
    if I.ndim == 3:
        I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    return I

# Colorise Functions
def ColoriseTerrain2D_ValueThresholdColorMapped(terrain, thresholdColors=[[]], defaultColor=[65, 105, 225]):
    color_world = np.ones((terrain.shape[0], terrain.shape[1], 3), np.uint8) * defaultColor
    for i in tqdm(range(terrain.shape[0])):
        for j in range(terrain.shape[1]):
            for th in thresholdColors:
                if terrain[i, j] >= th[0] and terrain[i, j] < th[1]:
                    color_world[i, j] = th[2]
    return color_world

def ColoriseTerrain2D_ArchipelagoSimple(terrain, beachThresh=[0.25, 0.6]):
    blue = [65, 105, 225]
    green = [34, 139, 34]
    beach = [238, 214, 175]
    color_world = ColoriseTerrain2D_ValueThresholdColorMapped(terrain, thresholdColors=[[beachThresh[0], beachThresh[1], beach], [beachThresh[1], 1.0, green]], defaultColor=blue)
    return color_world

# Driver Code
# Params

# Params

# RunCode