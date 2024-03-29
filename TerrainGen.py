'''
Python tool to generate 3D Terrains and Planets Procedurally
'''

# Imports
import cv2
import noise
import functools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Utils.Utils import *
from Utils.MeshLibrary import *

# Main Functions
def GeneratePerlinNoise_2D(WorldSize, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=0):
    world = np.zeros((WorldSize[0], WorldSize[1]))
    for i in tqdm(range(WorldSize[0]), disable=True):
        for j in range(WorldSize[1]):
            world[i][j] = noise.pnoise2(i/scale, j/scale,
            octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=repeatx, repeaty=repeaty, base=base)
    
    return world

def GeneratePerlinNoise_3D(WorldSize, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, repeatz=1024, base=0):
    world = np.zeros(WorldSize)
    for i in tqdm(range(WorldSize[0]), disable=True):
        for j in range(WorldSize[1]):
            for k in range(WorldSize[2]):
                world[i][j][k] = noise.pnoise3(i/scale, j/scale, k/scale,
                octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=repeatx, repeaty=repeaty, repeatz=repeatz, base=base)
    
    return world

def GeneratePerlinNoise_3D_From2D(WorldSize, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=[0, 1, 2]):
    vals = []
    for i in range(WorldSize[2]):
        v = GeneratePerlinNoise_2D(WorldSize, scale=scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=repeatx, repeaty=repeaty, base=base[i])
        vals.append(v)
    Noise = np.dstack(tuple(vals))
    return Noise

def main(WorldSize, Funcs, savePath, saveName, DepthScale=100, ExportDepthMultiplier=1, normalise=True, display=True, save=False, export3DModel=False):
    GenFunc, MaskFunc, ColoriseFunc, DepthFunc = Funcs['Gen'], Funcs['Mask'], Funcs['Colorise'], Funcs['Depth']

    print("Generating Values...")
    I_Noise = GenFunc(WorldSize)

    # Normalise values to [0, 1]
    if normalise:
        print("Normalising Values...")
        print("Value Range:", np.min(I_Noise), "to", np.max(I_Noise))
        I_Noise = (I_Noise - np.min(I_Noise)) / (np.max(I_Noise) - np.min(I_Noise))

    if save:
        cv2.imwrite(savePath + saveName + "_Values" + '.png', cv2.cvtColor((I_Noise*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    if display:
        plt.subplot(1, 3, 1)
        plt.imshow((I_Noise*255).astype(np.uint8), 'gray')
        # plt.show()

    if MaskFunc is not None:
        print("Masking Values...")
        I_Masked = MaskFunc(I_Noise)
        print("Masked Value Range:", np.min(I_Masked), "to", np.max(I_Masked))

        # Normalise values to [0, 1]
        if normalise:
            print("Normalising Masked Values...")
            print("Value Range:", np.min(I_Masked), "to", np.max(I_Masked))
            I_Masked = (I_Masked - np.min(I_Masked)) / (np.max(I_Masked) - np.min(I_Masked))

        if save:
            cv2.imwrite(savePath + saveName + "_Masked" + '.png', cv2.cvtColor((I_Masked*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        if display:
            plt.subplot(1, 3, 2)
            plt.imshow((I_Masked*255).astype(np.uint8), 'gray')
            # plt.show()

        I_Noise = I_Masked


    if ColoriseFunc is not None:
        print("Colorising Values...")
        I_Colorised = ColoriseFunc(I_Noise)
        I_Colorised = I_Colorised * 255

        if save:
            cv2.imwrite(savePath + saveName + "_Colorised" + '.png', cv2.cvtColor(I_Colorised.astype(np.uint8), cv2.COLOR_BGR2RGB))
        if display:
            plt.subplot(1, 3, 3)
            plt.imshow(I_Colorised.astype(np.uint8), 'gray')
            # plt.show()

    if export3DModel:
        print("Calculating Depths...")
        Depths = DepthFunc(I_Noise) * DepthScale

        print("Exporting...")
        # Texture = (I_Noise*255).astype(np.uint8)
        Texture = I_Colorised.astype(np.uint8)
        cv2.imwrite(savePath + saveName + '.png', cv2.cvtColor(Texture, cv2.COLOR_BGR2RGB))
        mesh = DepthImage_to_Terrain(Depths*ExportDepthMultiplier, I_Noise, savePath + saveName + '.png', name=saveName, exportPath=savePath + saveName + '.obj')
    plt.show()

# Driver Code
# # Params
# WorldSize = (512, 512)

# GenFunc = functools.partial(GeneratePerlinNoise_2D, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, base=7)

# MaskFunc = functools.partial(Mask_CircularSmooth, r=0.75, s1=2.0, s2=200)

# ColoriseFunc = functools.partial(ColoriseTerrain2D_ArchipelagoSimple, thresholds=[0.25, 0.4, 0.85, 0.95])

# DepthFunc = functools.partial(DepthFunc_GreyScaleDepth)
# DepthScale = 100
# ExportDepthMultiplier = 1

# savePath = 'GeneratedVisualisations/'
# saveName = 'Terrain_3'

# normalise = True
# display = False
# save = True
# export3DModel = True
# # Params

# # RunCode
# Funcs = {"Gen": GenFunc, "Mask": MaskFunc, "Colorise": ColoriseFunc, "Depth": DepthFunc}
# main(WorldSize, Funcs, savePath, saveName, DepthScale=DepthScale, ExportDepthMultiplier=ExportDepthMultiplier, normalise=normalise, display=display, save=save, export3DModel=export3DModel)