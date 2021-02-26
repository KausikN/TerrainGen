'''
UI script for ImageDatabase
'''

# Imports
from UIUtils import Py2UI

# Main Functions
def UIFunctionCodeParser(text):

    line = text.strip()

    funcname = "Utils." + line.split('(')[0].strip()
    paramText = '('.join(line.split('(')[1:]).rstrip(',')
    if paramText.endswith(')'):
        paramText = paramText[:-1]
    if not paramText.strip() == "":
        paramText = ", " + paramText
    parsedData = 'functools.partial(' + funcname + paramText + ')'

    return parsedData

def UITerrainGenModuleCodeParser(text):
    text = "TerrainGen." + text
    return text

def UIUtilsModuleCodeParser(text):
    text = "Utils." + text
    return text

def UITextSelectParser(text):
    text = "'" + text + "'"
    return text

# Driver Code
# Params
jsonPath = 'UIUtils/TerrainGenUI.json'

specialCodeProcessing = {"Generator": UITerrainGenModuleCodeParser, "Coloriser": UIUtilsModuleCodeParser, "DepthFunc": UIUtilsModuleCodeParser, "Mask": UIFunctionCodeParser}
# Params

# RunCode
Py2UI.JSON2UI(jsonPath, specialCodeProcessing)