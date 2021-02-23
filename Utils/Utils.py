'''
Utils for TerrainGen
'''

# Imports
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Main Functions
# Depth Functions
def DepthFunc_GreyScaleDepth(I, options=None):
    if I.ndim == 3:
        I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    return I

# Mask Functions
def Mask_Circular(I, r):
    r = int(r * min(I.shape[0], I.shape[1]))
    a, b = int(I.shape[0]/2), int(I.shape[1]/2)
    y, x = np.ogrid[-a:I.shape[0]-a, -b:I.shape[1]-b]
    mask = x**2+y**2 <= r**2

    I = I * mask
    return I

def Mask_CircularSmooth(I, r=0.5, s1=2.0, s2=20):
    a, b = int(I.shape[0]/2), int(I.shape[1]/2)

    circle_grad = np.zeros_like(I)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            distx = abs(i - a)
            disty = abs(j - b)
            dist = (distx*distx + disty*disty)**(0.5)
            circle_grad[i][j] = dist

    # get it between -1 and 1
    circle_grad = circle_grad / np.max(circle_grad)
    circle_grad -= r
    circle_grad *= s1
    circle_grad = -circle_grad
    # shrink gradient
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if circle_grad[i][j] > 0:
                circle_grad[i][j] *= s2
    # get it between 0 and 1
    circle_grad = circle_grad / np.max(circle_grad)

    I = I * circle_grad
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

def ColoriseTerrain2D_ArchipelagoSimple(terrain, thresholds=[0.25, 0.6, 0.85, 0.95]):
    blue = [65, 105, 225]
    beach = [238, 214, 175]
    green = [34, 139, 34]
    mountain = [139, 137, 137]
    snow = [255, 250, 250]
    color_world = ColoriseTerrain2D_ValueThresholdColorMapped(terrain,
        thresholdColors=[
            [thresholds[0], thresholds[1], beach],
            [thresholds[1], thresholds[2], green],
            [thresholds[2], thresholds[3], mountain],
            [thresholds[3], 1.0, snow]
        ], 
        defaultColor=blue)
    return color_world

# Driver Code
# Params

# Params

# RunCode