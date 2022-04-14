'''
Mask Functions
'''

# Imports
import numpy as np

# Main Functions
def Mask_Nothing(I, **params):
    return I

def Mask_Circular(I, r=0.3, **params):
    '''
    Mask a 2D image with a hard thresholded circular mask
    '''
    # Get Params
    r = int(r * min(I.shape[0], I.shape[1]))
    # Construct Mask
    a, b = int(I.shape[0]/2), int(I.shape[1]/2)
    y, x = np.ogrid[-a:I.shape[0]-a, -b:I.shape[1]-b]
    mask = x**2+y**2 <= r**2
    # Apply Mask
    I = I * mask

    return I

def Mask_CircularSmooth(I, r=0.5, s1=2.0, s2=20, **params):
    '''
    Mask a 2D image with a smooth circular mask
    '''
    # Get Params
    a, b = int(I.shape[0]/2), int(I.shape[1]/2)
    # Construct Mask
    circle_grad = np.zeros_like(I)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            distx = abs(i - a)
            disty = abs(j - b)
            dist = (distx*distx + disty*disty)**(0.5)
            circle_grad[i][j] = dist
    # Normalise
    circle_grad = circle_grad / np.max(circle_grad)
    circle_grad -= r
    circle_grad *= s1
    circle_grad = -circle_grad
    # Shrink Grads
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if circle_grad[i][j] > 0:
                circle_grad[i][j] *= s2
    # Normalise
    circle_grad = circle_grad / np.max(circle_grad)
    # Apply Mask
    I = I * circle_grad

    return I

# Main Vars
MASK_FUNCS = {
    "Nothing": Mask_Nothing,
    "Circular": Mask_Circular,
    "Circular Smooth": Mask_CircularSmooth
}