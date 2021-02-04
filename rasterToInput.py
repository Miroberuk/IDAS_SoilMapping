import copy
import sys
import numpy as np
import glob
import os.path
from skimage import io

##############################################################
def CropToTile (im, size):
    if len(im.shape) == 2:#handle greyscale
        im = im.reshape(im.shape[0], im.shape[1],1)

    crop_dim0 = size * (im.shape[0]//size)
    crop_dim1 = size * (im.shape[1]//size)
    return im[0:crop_dim0, 0:crop_dim1, :]

def SlideRastersToTiles(im, CLS, size):

    if len(im.shape) ==2:
        h, w = im.shape
        d = 1
    else:
        h, w, d = im.shape

    TileTensor = np.zeros(((h-size)*(w-size), size,size,d))
    Label = np.zeros(((h-size)*(w-size),1))
    B=0
    for y in range(0, h-size):
        for x in range(0, w-size):
            Label[B] = np.median(CLS[y:y+size,x:x+size].reshape(1,-1))

            TileTensor[B,:,:,:] = im[y:y+size,x:x+size,:].reshape(size,size,d)
            B+=1

    return TileTensor, Label

#Create the label vector
def PrepareTensorData(ImageTile, ClassTile, size):
    #this takes the image tile tensor and the class tile tensor
    #It produces a label vector from the tiles which have 90% of a pure class
    #It then extracts the image tiles that have a classification value in the labels
    LabelVector = np.zeros(ClassTile.shape[0])

    for v in range(0,ClassTile.shape[0]):
        Tile = ClassTile[v,:,:,0]
        vals, counts = np.unique(Tile, return_counts = True)
        if (vals[0] == 0) and (counts[0] > 0.1 * size**2):
            LabelVector[v] = 0
        elif counts[np.argmax(counts)] >= 0.9 * size**2:
            LabelVector[v] = vals[np.argmax(counts)]

    LabelVector = LabelVector[LabelVector > 0]
    ClassifiedTiles = np.zeros((np.count_nonzero(LabelVector), size,size,3))
    C = 0
    for t in range(0,np.count_nonzero(LabelVector)):
        if LabelVector[t] > 0:
            ClassifiedTiles[C,:,:,:] = ImageTile[t,:,:,:]
            C += 1
    return LabelVector, ClassifiedTiles


##
#TODO running locally, fix directory pathing
size = 50
img = glob.glob("*.jpg")
TestTuple = []
for im in img:
    TestTuple.append(os.path.basename(im).partition('_')[0])
TestTuple = np.unique(TestTuple)
class_img = glob.glob("SCLS_*.tif*")

for f, satellite in enumerate(TestTuple):
    for i, im in enumerate(img):
        print('im is ' + os.path.basename(im))
        Im3D = io.imread(im)
        if len(Im3D) == 2:
            Im3D = Im3D[0]
        Class = io.imread(class_img[i], as_gray=True)
        if (Class.shape[0] != Im3D.shape[0]) or (Class.shape[1] != Im3D.shape[1]):
            print('WARNING: inconsistent image and class mask sizes for ' + im)
            Class = T.resize(Class, (Im3D.shape[0], Im3D.shape[1]), preserve_range = True) #bug handling for vector
        ClassIm = copy.deepcopy(Class)
        (TileTensor, labels) = slide_rasters_to_tiles(im,Class,size)
