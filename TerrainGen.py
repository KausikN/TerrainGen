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

from Utils import Utils
from Utils import MeshLibrary

# Main Functions
def GeneratePerlinNoise_2D(WorldSize, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=0):
    world = np.zeros((WorldSize[0], WorldSize[1]))
    for i in tqdm(range(WorldSize[0])):
        for j in range(WorldSize[1]):
            world[i][j] = noise.pnoise2(i/scale, j/scale,
            octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=repeatx, repeaty=repeaty, base=base)
    
    return world

def GeneratePerlinNoise_3D(WorldSize, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, repeatz=1024, base=0):
    world = np.zeros(WorldSize)
    for i in tqdm(range(WorldSize[0])):
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

# Driver Code
# Params
WorldSize = (256, 256)

GenFunc = functools.partial(GeneratePerlinNoise_2D, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0, base=0)

DepthFunc = Utils.DepthFunc_GreyScaleDepth
DepthScale = 100
ExportDepthMultiplier = 1

savePath = 'GeneratedVisualisations/'
saveName = 'Terrain_1'

display = True
save = True
# Params

# RunCode
I_Noise = GenFunc(WorldSize)

if display:
    plt.imshow((I_Noise*255).astype(np.uint8), 'gray')
    plt.show()

if save:
    print("Calculating Depths...")
    Depths = DepthFunc(I_Noise) * DepthScale



    print("Exporting...")
    Texture = (I_Noise*255).astype(np.uint8)
    cv2.imwrite(savePath + saveName + '.png', Texture)
    mesh = MeshLibrary.DepthImage_to_Terrain(Depths*ExportDepthMultiplier, I_Noise, savePath + saveName + '.png', name=saveName, exportPath=savePath + saveName + '.obj')