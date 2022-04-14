'''
Functions for 3D Processing
'''

# Imports
import os
import cv2
import open3d
import numpy as np

# Main Functions
def DepthImage_to_Terrain(depths, I, ImagePath, name='Test', exportPath=None):
    '''
    Converts Depth Image to 3D Mesh
    '''
    vertexTextureUVs = []
    vertexNormals = [[0.0, 1.0, 0.0]]
    
    vertices = []
    for i in range(depths.shape[0]):
        for j in range(depths.shape[1]):
            vertices.append([j+1, i+1, depths[-i-1, j]])
            vertexTextureUVs.append([j/depths.shape[1], i/depths.shape[0]])

    faceMaps = []
    for i in range(depths.shape[0]-1):
        for j in range(depths.shape[1]-1):
            faceMaps.append([
                [(i*depths.shape[1]) + j +1]*3,
                [((i+1)*depths.shape[1]) + j +1]*3,
                [((i+1)*depths.shape[1]) + j+1 +1]*3,
                [(i*depths.shape[1]) + j+1 +1]*3
            ])

    mesh = Object3D(name, vertices, faceMaps, ImagePath, vertexTextureUVs, vertexNormals)
    
    if not (exportPath is None):
        # Initial Write Without Normals
        OBJWrite(mesh, exportPath)
        MtlWrite(mesh, exportPath.rstrip('.obj') + ".mtl")

        # Reread and Compute Normals and ReExport
        tri_mesh = open3d.io.read_triangle_mesh(exportPath)
        tri_mesh.compute_vertex_normals()

        vertexNormals = np.asarray(tri_mesh.vertex_normals)
        mesh.vertexNormals = list(vertexNormals)

        # open3d.visualization.draw_geometries([tri_mesh])
        # open3d.io.write_triangle_mesh(exportPath, tri_mesh, compressed=True)
        OBJWrite(mesh, exportPath)
    
    return mesh

# OBJ Functions
class Object3D:
    '''
    3D Object
    '''
    def __init__(self, name, vertices, faceMaps, imgPath="", vertexTextureUVs=None, vertexNormals=None):
        self.name = name
        self.imgPath = imgPath
        self.vertices = vertices
        self.faceMaps = faceMaps
        self.vertexTextureUVs = vertexTextureUVs
        self.vertexNormals = vertexNormals
        if self.vertexTextureUVs is None:
            self.vertexTextureUVs = [[0.0, 0.0, 0.0]]
        if self.vertexNormals is None:
            self.vertexNormals = [[0.0, 0.0, 0.0]]

def OBJWrite(obj, savePath):
    '''
    Writes OBJ File with the given 3D Object data
    '''
    """
    # Blender v2.81 (sub 16) OBJ File: ''
    # www.blender.org
    mtllib Cube.mtl
    o Cube
    v 0.000000 0.000000 0.000000
    v 1.000000 0.000000 0.000000
    v 1.000000 1.000000 0.000000
    v 0.000000 1.000000 0.000000
    vt 0.625000 0.500000
    vt 0.875000 0.500000
    vt 0.875000 0.750000
    vt 0.625000 0.750000
    vn 0.0000 1.0000 0.0000
    vn 0.0000 0.0000 1.0000
    vn -1.0000 0.0000 0.0000
    vn 0.0000 -1.0000 0.0000
    usemtl Material
    s off
    f 1/1/1 2/2/2 3/3/3 4/4/4
    """
    filename = os.path.splitext(os.path.basename(savePath))[0]

    Header = [
        "# Blender v2.81 (sub 16) OBJ File: ''",
        '# www.blender.org'
    ]

    MtlLib = 'mtllib ' + filename + '.mtl'

    Name = 'o ' + obj.name

    Vertices = []
    for v in obj.vertices:
        Vertices.append('v ' + Get6DecFloatString(v[0]) + ' ' + Get6DecFloatString(v[1]) + ' ' + Get6DecFloatString(v[2]))
    
    VertexTextureUVs = []
    for v in obj.vertexTextureUVs:
        VertexTextureUVs.append('vt ' + Get6DecFloatString(v[0]) + ' ' + Get6DecFloatString(v[1]))
    
    VertexNormals = []
    for v in obj.vertexNormals:
        VertexNormals.append('vn ' + Get6DecFloatString(v[0]) + ' ' + Get6DecFloatString(v[1]) + ' ' + Get6DecFloatString(v[2]))
    
    MidText = [
        'usemtl ' + obj.name,
        's off'
    ]

    Faces = []
    for f in obj.faceMaps:
        fline = 'f '
        for v in f:
            fline = fline + '' + str(v[0]) + '/' + str(v[1]) + '/' + str(v[2]) + ' '
        Faces.append(fline.rstrip())
    
    OBJTextLines = list(Header)
    OBJTextLines.append(MtlLib)
    OBJTextLines.append(Name)
    OBJTextLines.extend(Vertices)
    OBJTextLines.extend(VertexTextureUVs)
    OBJTextLines.extend(VertexNormals)
    OBJTextLines.extend(MidText)
    OBJTextLines.extend(Faces)

    open(savePath, 'w').write('\n'.join(OBJTextLines))

def MtlWrite(obj, savePath):
    '''
    Writes MTL File with the given 3D Object data
    '''
    """
    # Blender MTL File: 'None'
    # Material Count: 1

    newmtl iiit
    Ka 1.000 1.000 1.000
    Kd 1.000 1.000 1.000
    Ks 0.000 0.000 0.000
    map_Kd iiit.png
    """

    Header = [
        "# Blender MTL File: 'None'",
        '# Material Count: 1',
        ''
    ]

    MtlLine = 'newmtl ' + obj.name

    MtlData = [
        "Ns 225.000000",
        "Ka 1.000 1.000 1.000",
        "Kd 1.000 1.000 1.000",
        "Ks 0.000 0.000 0.000"
        ]

    TexFileLine = "map_Kd " + os.path.basename(obj.imgPath)

    MtlTextLines = list(Header)
    MtlTextLines.append(MtlLine)
    MtlTextLines.extend(MtlData)
    MtlTextLines.append(TexFileLine)

    open(savePath, 'w').write('\n'.join(MtlTextLines))

def Get6DecFloatString(val):
    '''
    Converts a float to a string with 6 decimal places
    '''
    return str(round(float(val), 6))
    val = round(float(val), 6)
    # print(val)
    decCount = len(str(val).split('.')[1])
    extraPadding = 6 - decCount
    strval = str(val) + ('0'*extraPadding)
    return strval

# Driver Code