"""
Stream lit GUI for hosting TerrainGen
"""

# Imports
import numpy as np
import streamlit as st
import json
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg

from TerrainGen import *

# Main Vars
config = json.load(open('./StreamLitGUI/UIConfig.json', 'r'))

# Main Functions
def main():
    # Create Sidebar
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
        tuple(
            [config['PROJECT_NAME']] + 
            config['PROJECT_MODES']
        )
    )
    
    if selected_box == config['PROJECT_NAME']:
        HomePage()
    else:
        correspondingFuncName = selected_box.replace(' ', '_').lower()
        if correspondingFuncName in globals().keys():
            globals()[correspondingFuncName]()
 

def HomePage():
    st.title(config['PROJECT_NAME'])
    st.markdown('Github Repo: ' + "[" + config['PROJECT_LINK'] + "](" + config['PROJECT_LINK'] + ")")
    st.markdown('\n'.join(config['PROJECT_DESC']))

    # st.write(open(config['PROJECT_README'], 'r').read())

#############################################################################################################################
# Repo Based Vars
WORLDSIZE_MIN = [1, 1]
WORLDSIZE_MAX = [2048, 2048]
WORLDSIZE_DEFAULT = [100, 100]
COLORISEDIMAGE_SAVEPATH_DEFAULT = "TerrainGen3DLandscape.png"
LANDSCAPEMODEL_SAVEPATH_DEFAULT = "TerrainGen3DLandscape"

DEFAULT_COLORISER_BASEGRADIENT = {
    "start": [0.0, 0.0, 0.0],
    "end": [0.25, 0.41, 0.88]
}
DEFAULT_COLORISER_THRESHOLDS = [0.25, 0.6, 0.85, 0.95]
DEFAULT_COLORISER_COLORS = [
    [0.93, 0.84, 0.69],
    [0.13, 0.55, 0.13],
    [0.55, 0.54, 0.54],
    [1.0, 0.98, 0.98]
]

# Util Vars
WORLDSIZEINDICATORIMAGE_SIZE = [128, 128]
COLORISERINDICATORIMAGE_SIZE = [128, 128]
COLORISERINDICATOR_SAVEPATH_DEFAULT = "INDICATOR.png"
PLOTINDICATORIMAGE_SIZE = [256, 256]

RADIALINDICATORIMAGE = None

# Util Functions
def Hex_to_RGB(val):
    val = val.lstrip('#')
    lv = len(val)
    rgb = tuple(int(val[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    rgb = np.array(rgb) / 255.0
    return rgb

def RGB_to_Hex(rgb):
    rgb = np.array(np.array(rgb) * 255, dtype=np.uint8)
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

@st.cache
def GenerateRadialIndicatorImage():
    global RADIALINDICATORIMAGE
    if RADIALINDICATORIMAGE is None:
        # Generate Radial vals
        x_vals = np.linspace(-1.0, 1.0, COLORISERINDICATORIMAGE_SIZE[0])[:, None]
        y_vals = np.linspace(-1.0, 1.0, COLORISERINDICATORIMAGE_SIZE[1])[None, :]
        RADIALINDICATORIMAGE = np.sqrt(x_vals ** 2 + y_vals ** 2)
        RADIALINDICATORIMAGE = 1 - ((RADIALINDICATORIMAGE - np.min(RADIALINDICATORIMAGE)) / (np.max(RADIALINDICATORIMAGE) - np.min(RADIALINDICATORIMAGE)))
    
    return RADIALINDICATORIMAGE

def GeneratePlotDepthsIndicatorImage(USERINPUT_DepthScale):
    global RADIALINDICATORIMAGE
    RADIALINDICATORIMAGE = GenerateRadialIndicatorImage()
    I_i = np.array(RADIALINDICATORIMAGE*255, dtype=np.uint8)
    plottedDepthsFigure = PlotImage3D_Plane(I_i, RADIALINDICATORIMAGE*USERINPUT_DepthScale, display=False)

    return plottedDepthsFigure

@st.cache
def GenerateColoriserIndicatorImage(USERINPUT_BaseGradient, USERINPUT_Thresholds, USERINPUT_ThresholdColors):
    RADIALINDICATORIMAGE = GenerateRadialIndicatorImage()
    # Get Intervals
    Intervals = []
    prevVal, prevColor = 0.0, USERINPUT_BaseGradient["start"]
    for i in range(len(USERINPUT_Thresholds)):
        Intervals.append({
            "interval": [prevVal, USERINPUT_Thresholds[i]],
            "gradient": {
                "start": prevColor,
                "end": USERINPUT_ThresholdColors[i]
            }
        })
        prevVal = USERINPUT_Thresholds[i]
        prevColor = USERINPUT_ThresholdColors[i]
    Intervals.append({
        "interval": [prevVal, 1.0],
        "gradient": {
            "start": prevColor,
            "end": USERINPUT_BaseGradient["end"]
        }
    })
    # Get colorised indicator image
    I_ic = ColoriseImage_Thresholding_RangedGradients(RADIALINDICATORIMAGE, Intervals)

    return I_ic

@st.cache
def GenerateWorldSizeIndicatorImage(WorldSize):
    # World Size Indicator Image 
    WorldSizeIndicator_Image = np.zeros((WORLDSIZEINDICATORIMAGE_SIZE[0], WORLDSIZEINDICATORIMAGE_SIZE[1]), dtype=int)
    WorldSizeIndicator_Image[:int((WorldSize[0]/WORLDSIZE_MAX[0])*WORLDSIZEINDICATORIMAGE_SIZE[0]), :int((WorldSize[1]/WORLDSIZE_MAX[1])*WORLDSIZEINDICATORIMAGE_SIZE[1])] = 255
    
    return WorldSizeIndicator_Image

@st.cache
def GenerateNoiseImage2D(
    WorldSize, 
    USERINPUT_Scale, USERINPUT_Octaves, USERINPUT_Persistence, USERINPUT_Lacunarity, USERINPUT_RepeatX, USERINPUT_RepeatY, USERINPUT_Seed
    ):
    # Noise Image 2D
    USERINPUT_Scale = max(1, USERINPUT_Scale)
    USERINPUT_RepeatX, USERINPUT_RepeatY = max(1, USERINPUT_RepeatX), max(1, USERINPUT_RepeatY)
    Noise = GeneratePerlinNoise_2D(
        WorldSize, 
        scale=USERINPUT_Scale, octaves=USERINPUT_Octaves, persistence=USERINPUT_Persistence, 
        lacunarity=USERINPUT_Lacunarity, repeatx=USERINPUT_RepeatX, repeaty=USERINPUT_RepeatY, 
        base=USERINPUT_Seed
    )

    return Noise

@st.cache
def Generate2DTerrain(
    WorldSize, USERINPUT_Scale, USERINPUT_Octaves, USERINPUT_Persistence, USERINPUT_Lacunarity, USERINPUT_RepeatX, USERINPUT_RepeatY, USERINPUT_Seed, 
    USERINPUT_MaskFunc, 
    USERINPUT_Thresholds, USERINPUT_ThresholdColors, USERINPUT_BaseGradient
    ):
    # Colorised Terrain Image
    # Generate Noise and normalise
    Noise = GenerateNoiseImage2D(WorldSize, USERINPUT_Scale, USERINPUT_Octaves, USERINPUT_Persistence, USERINPUT_Lacunarity, USERINPUT_RepeatX, USERINPUT_RepeatY, USERINPUT_Seed)
    I_Noise = (Noise - np.min(Noise)) / (np.max(Noise) - np.min(Noise))
    # Mask
    I_Noise = USERINPUT_MaskFunc(I_Noise)
    I_Noise = (I_Noise - np.min(I_Noise)) / (np.max(I_Noise) - np.min(I_Noise))
    # Colorise
    # Get Intervals
    Intervals = []
    prevVal, prevColor = 0.0, USERINPUT_BaseGradient["start"]
    for i in range(len(USERINPUT_Thresholds)):
        Intervals.append({
            "interval": [prevVal, USERINPUT_Thresholds[i]],
            "gradient": {
                "start": prevColor,
                "end": USERINPUT_ThresholdColors[i]
            }
        })
        prevVal = USERINPUT_Thresholds[i]
        prevColor = USERINPUT_ThresholdColors[i]
    Intervals.append({
        "interval": [prevVal, 1.0],
        "gradient": {
            "start": prevColor,
            "end": USERINPUT_BaseGradient["end"]
        }
    })
    # Get colorised indicator image
    I_Colorised = ColoriseImage_Thresholding_RangedGradients(I_Noise, Intervals)
    
    return I_Colorised

def Generate3DLanscape(
    WorldSize, USERINPUT_Scale, USERINPUT_Octaves, USERINPUT_Persistence, USERINPUT_Lacunarity, USERINPUT_RepeatX, USERINPUT_RepeatY, USERINPUT_Seed, 
    USERINPUT_MaskFunc, 
    USERINPUT_Thresholds, USERINPUT_ThresholdColors, USERINPUT_BaseGradient, 
    USERINPUT_DepthScale
    ):
    # 3D Landscape
    # Generate Noise and normalise
    Noise = GenerateNoiseImage2D(WorldSize, USERINPUT_Scale, USERINPUT_Octaves, USERINPUT_Persistence, USERINPUT_Lacunarity, USERINPUT_RepeatX, USERINPUT_RepeatY, USERINPUT_Seed)
    I_Noise = (Noise - np.min(Noise)) / (np.max(Noise) - np.min(Noise))
    # Mask
    I_Noise = USERINPUT_MaskFunc(I_Noise)
    I_Noise = (I_Noise - np.min(I_Noise)) / (np.max(I_Noise) - np.min(I_Noise))
    # Colorise
    # Get Intervals
    Intervals = []
    prevVal, prevColor = 0.0, USERINPUT_BaseGradient["start"]
    for i in range(len(USERINPUT_Thresholds)):
        Intervals.append({
            "interval": [prevVal, USERINPUT_Thresholds[i]],
            "gradient": {
                "start": prevColor,
                "end": USERINPUT_ThresholdColors[i]
            }
        })
        prevVal = USERINPUT_Thresholds[i]
        prevColor = USERINPUT_ThresholdColors[i]
    Intervals.append({
        "interval": [prevVal, 1.0],
        "gradient": {
            "start": prevColor,
            "end": USERINPUT_BaseGradient["end"]
        }
    })
    # Get colorised indicator image
    I_Colorised = ColoriseImage_Thresholding_RangedGradients(I_Noise, Intervals)
    # Generate Mesh and Save
    I_Colorised_i = np.array(I_Colorised * 255, dtype=np.uint8)
    I_Colorised_i = cv2.cvtColor(I_Colorised_i, cv2.COLOR_RGB2BGR)
    plottedDepthsFigure = PlotImage3D_Plane(I_Colorised_i, I_Noise*USERINPUT_DepthScale, display=False)
    # MeshLibrary.DepthImage_to_Terrain(I_Noise*USERINPUT_DepthScale, None, COLORISEDIMAGE_SAVEPATH_DEFAULT, name=LANDSCAPEMODEL_SAVEPATH_DEFAULT, exportPath=LANDSCAPEMODEL_SAVEPATH_DEFAULT + '.obj')

    return plottedDepthsFigure, I_Colorised

# UI Utils Functions
def UI_WorldSizeParams():
    # Inputs
    USERINPUT_WorldSizeY = st.sidebar.slider("Width Pixels", WORLDSIZE_MIN[0], WORLDSIZE_MAX[0], WORLDSIZE_DEFAULT[0], WORLDSIZE_MIN[0], key="USERINPUT_WorldSizeX")
    USERINPUT_WorldSizeX = st.sidebar.slider("Height Pixels", WORLDSIZE_MIN[1], WORLDSIZE_MAX[1], WORLDSIZE_DEFAULT[1], WORLDSIZE_MIN[1], key="USERINPUT_WorldSizeY")
    WorldSize = [int(USERINPUT_WorldSizeX), int(USERINPUT_WorldSizeY)]
    # Display
    WorldSizeIndicator_Image = GenerateWorldSizeIndicatorImage(WorldSize)
    st.sidebar.image(WorldSizeIndicator_Image, caption="World Size (Max " + str(WORLDSIZE_MAX[0]) + " x " + str(WORLDSIZE_MAX[1]) + ")", use_column_width=False, clamp=False)
    
    return WorldSize

def UI_GenNoiseParams():
    # Inputs
    USERINPUT_Seed = st.slider("Seed", 0, 400, 0, 1, key="USERINPUT_Seed")
    col1, col2, col3 = st.columns(3)
    USERINPUT_Scale = col1.slider("Scale", 0, 500, 100, 50, key="USERINPUT_Scale")
    USERINPUT_Octaves = col2.slider("Octaves", 1, 12, 6, 1, key="USERINPUT_Octaves")
    USERINPUT_Persistence = col3.slider("Persistence", 0.0, 1.0, 0.5, 0.1, key="USERINPUT_Persistence")
    USERINPUT_Lacunarity = col1.slider("Lacunarity", 1, 20, 2, 1, key="USERINPUT_Lacunarity")
    USERINPUT_RepeatX = col2.slider("X Bounds", 0, 2048, 1024, 128, key="USERINPUT_RepeatX")
    USERINPUT_RepeatY = col3.slider("Y Bounds", 0, 2048, 1024, 128, key="USERINPUT_RepeatY")

    return USERINPUT_Scale, USERINPUT_Octaves, USERINPUT_Persistence, USERINPUT_Lacunarity, USERINPUT_RepeatX, USERINPUT_RepeatY, USERINPUT_Seed

def UI_MaskParams():
    # Inputs
    USERINPUT_Mask = st.selectbox("Mask Function", list(MASK_FUNCS.keys()))
    USERINPUT_MaskFunc = MASK_FUNCS[USERINPUT_Mask]
    
    return USERINPUT_MaskFunc

def UI_Coloriser():
    # Inputs
    USERINPUT_ColorCount = st.slider("Terrain Type Count", 2, 10, step=1, value=5)
    col1, indIcol = st.columns(2)
    USERINPUT_BaseGradient = {
        "start": list(Hex_to_RGB(col1.color_picker("Select Base Gradient Start", value=RGB_to_Hex(DEFAULT_COLORISER_BASEGRADIENT["start"]))).astype(np.float32)),
        "end": list(Hex_to_RGB(col1.color_picker("Select Base Gradient End", value=RGB_to_Hex(DEFAULT_COLORISER_BASEGRADIENT["end"]))))
    }
    USERINPUT_Thresholds = []
    USERINPUT_ThresholdColors = []
    for i in range(USERINPUT_ColorCount-1):
        col1, col2 = st.columns(2)
        color = list(Hex_to_RGB(col1.color_picker("Terrain #" + str(i+1) + " color", value=RGB_to_Hex(DEFAULT_COLORISER_COLORS[i % len(DEFAULT_COLORISER_COLORS)]))))
        th = col2.slider("Terrain #" + str(i+1) + " minimum threshold", 0.0, 1.0, DEFAULT_COLORISER_THRESHOLDS[i % len(DEFAULT_COLORISER_THRESHOLDS)], 0.05)
        USERINPUT_Thresholds.append(th)
        USERINPUT_ThresholdColors.append(color)
    # Display
    I_IndicatorColorised = GenerateColoriserIndicatorImage(USERINPUT_BaseGradient, USERINPUT_Thresholds, USERINPUT_ThresholdColors)
    indIcol.image(I_IndicatorColorised, caption="Coloriser Indicator Image", use_column_width=False, clamp=True)

    return USERINPUT_BaseGradient, USERINPUT_Thresholds, USERINPUT_ThresholdColors

def UI_3DDepth():
    # Inputs
    USERINPUT_DepthScale = st.slider("Depth Scale", 0.0, 10.0, step=0.1, value=1.0)
    # Display
    depthPlotFig = GeneratePlotDepthsIndicatorImage(USERINPUT_DepthScale)
    st.plotly_chart(depthPlotFig, use_container_width=True)

    return USERINPUT_DepthScale

# Repo Based Functions
def generate_2d_perlin_noise():
    # Title
    st.header("Generate 2D Perlin Noise")

    # Load Inputs
    st.write("## Generation Parameters")
    ## World Size
    WorldSize = UI_WorldSizeParams()
    ## Other Gen Params
    USERINPUT_Scale, USERINPUT_Octaves, USERINPUT_Persistence, USERINPUT_Lacunarity, USERINPUT_RepeatX, USERINPUT_RepeatY, USERINPUT_Seed = UI_GenNoiseParams()

    # Process Inputs on Button Click
    if st.button("Generate"):
        Noise = GenerateNoiseImage2D(WorldSize, USERINPUT_Scale, USERINPUT_Octaves, USERINPUT_Persistence, USERINPUT_Lacunarity, USERINPUT_RepeatX, USERINPUT_RepeatY, USERINPUT_Seed)
        I_Noise = (Noise - np.min(Noise)) / (np.max(Noise) - np.min(Noise))

        # Display Outputs
        st.image(I_Noise, caption="Noise Image", use_column_width=False, clamp=True)

def generate_terrain():
    # Title
    st.header("Generate Terrain")

    # Load Inputs
    ## Generation Params
    st.write("## Generation Parameters")
    ### World Size Params
    WorldSize = UI_WorldSizeParams()
    ### Other Gen Params
    USERINPUT_Scale, USERINPUT_Octaves, USERINPUT_Persistence, USERINPUT_Lacunarity, USERINPUT_RepeatX, USERINPUT_RepeatY, USERINPUT_Seed = UI_GenNoiseParams()
    ## Mask Params
    st.write("## Mask Parameters")
    USERINPUT_MaskFunc = UI_MaskParams()
    ## Colourisation Params
    st.write("## Colouring Parameters")
    USERINPUT_BaseGradient, USERINPUT_Thresholds, USERINPUT_ThresholdColors = UI_Coloriser()

    # Process Inputs on Button Click
    if st.button("Generate"):
        I_Terrain = Generate2DTerrain(
            WorldSize, 
            USERINPUT_Scale, USERINPUT_Octaves, USERINPUT_Persistence, USERINPUT_Lacunarity, USERINPUT_RepeatX, USERINPUT_RepeatY, USERINPUT_Seed, 
            USERINPUT_MaskFunc, 
            USERINPUT_Thresholds, USERINPUT_ThresholdColors, USERINPUT_BaseGradient
        )

        # Display Outputs
        st.image(I_Terrain, caption="Terrain", use_column_width=False, clamp=True)

def generate_3d_landscape():
    # Title
    st.header("Generate 3D Landscape")

    # Load Inputs
    ## Generation Params
    st.write("## Generation Parameters")
    ### World Size Params - FOr Landscapes max size is kept lower for memory efficiency
    global WORLDSIZE_MAX
    global WORLDSIZE_DEFAULT
    WORLDSIZE_MAX = [50, 50]
    WORLDSIZE_DEFAULT = [10, 10]
    WorldSize = UI_WorldSizeParams()
    ### Other Gen Params
    USERINPUT_Scale, USERINPUT_Octaves, USERINPUT_Persistence, USERINPUT_Lacunarity, USERINPUT_RepeatX, USERINPUT_RepeatY, USERINPUT_Seed = UI_GenNoiseParams()
    ## Mask Params
    st.write("## Mask Parameters")
    USERINPUT_MaskFunc = UI_MaskParams()
    ## Colourisation Params
    st.write("## Colouring Parameters")
    USERINPUT_BaseGradient, USERINPUT_Thresholds, USERINPUT_ThresholdColors = UI_Coloriser()
    ## Depth Params
    st.write("## 3D Parameters")
    USERINPUT_DepthScale = UI_3DDepth()

    # Process Inputs on Button Click
    if st.button("Generate"):
        landscapePlotFig, I_Noise = Generate3DLanscape(
            WorldSize, 
            USERINPUT_Scale, USERINPUT_Octaves, USERINPUT_Persistence, USERINPUT_Lacunarity, USERINPUT_RepeatX, USERINPUT_RepeatY, USERINPUT_Seed, 
            USERINPUT_MaskFunc, 
            USERINPUT_Thresholds, USERINPUT_ThresholdColors, USERINPUT_BaseGradient, 
            USERINPUT_DepthScale
        )
        # Display Outputs
        st.image(I_Noise, caption="Noise Image")
        st.plotly_chart(landscapePlotFig, use_container_width=True)

#############################################################################################################################
# Driver Code
if __name__ == "__main__":
    main()