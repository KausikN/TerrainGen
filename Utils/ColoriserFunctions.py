'''
Coloriser Functions
'''

# Imports
import numpy as np

# Main Functions
# Generic Colorise Function
def ColoriseImage_Thresholding_RangedGradients(I, intervals=[], **params):
    '''
    Colorise Image based on Ranged Interval Gradients
    '''
    # Init color image
    I_c = np.zeros((I.shape[0], I.shape[1], 3), dtype=float)
    # For each interval
    for intData in intervals:
        # Get interval data
        interval = intData["interval"]
        colorGrad_start, colorGrad_end = np.array(intData["gradient"]["start"]), np.array(intData["gradient"]["end"])
        # Mask and colorize
        mask = (I >= interval[0]) & (I < interval[1])
        maskedVals = np.dstack((I[mask], I[mask], I[mask]))
        I_c[mask] = colorGrad_start + (colorGrad_end - colorGrad_start) * maskedVals

    return I_c

# Specific Coloriser Functions
def ColoriseTerrain2D_ArchipelagoSimple(terrain, thresholds=[0.25, 0.6, 0.85, 0.95]):
    '''
    Colorise Terrain 2D based on Archipelago Colors
    '''
    # Define Colors
    white = [1.0, 1.0, 1.0] # [255, 255, 255]
    blue = [0.25, 0.6, 0.85] # [65, 105, 225]
    beach = [0.93, 0.84, 0.69] # [238, 214, 175]
    green = [0.13, 0.55, 0.13] # [34, 139, 34]
    mountain = [0.55, 0.54, 0.54] # [139, 137, 137]
    snow = [1.0, 0.98, 0.98] # [255, 250, 250]
    thresholdColors = [beach, green, mountain, snow]
    # Form Intervals
    Intervals = []
    prevVal, prevColor = 0.0, blue
    for i in range(len(thresholds)):
        Intervals.append({
            "interval": [prevVal, thresholds[i]],
            "gradient": {
                "start": prevColor,
                "end": thresholdColors[i]
            }
        })
        prevVal = thresholds[i]
        prevColor = thresholdColors[i]
    Intervals.append({
        "interval": [prevVal, 1.0],
        "gradient": {
            "start": prevColor,
            "end": white
        }
    })
    # Colorise
    color_world = ColoriseImage_Thresholding_RangedGradients(terrain, intervals=Intervals)

    return color_world

# Main Vars
COLORISER_FUNCS = {
    "Ranged Gradients": ColoriseImage_Thresholding_RangedGradients
}

COLORISER_SPECIFIC_FUNCS = {
    "Terrain 2D Archipelago Simple": ColoriseTerrain2D_ArchipelagoSimple
}