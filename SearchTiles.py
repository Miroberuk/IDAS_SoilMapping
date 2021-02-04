
import numpy as np
import os
import pandas as pd
import skimage.io as IO
import sys
import warnings
import tifffile
from PIL import Image

arr = []
#OUTPUT = 'D:\\Users\\mirob_uhy4ay7\\Documents\\IDAS Project\\InputFiles\\C1'
ClassNum = 'Class1'
base_dir = 'D:/SoilSampleProject/00_IDaS/2020_July_Sentinel/Dataset/' + ClassNum
tiles = []
for i in os.listdir(base_dir):
    tileNumStart = (i.find("Tile_")+5)
    tileNumEnd = (i.find("_Band"))
    tileNum = i[tileNumStart:tileNumEnd]
    tiles.append(tileNum)
    # if (i.find("Band12") != -1):
    #     arr.append(i)

myTiles = list(dict.fromkeys(tiles))
index = myTiles[0]
for i in os.listdir(base_dir):
    if (i.find(index) != -1):
        arr.append(i)

Bands = ['Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8', 'Band11', 'Band12']

# for tileNum in myTiles:
#     for bandNum in Bands:
#         tileLocation = base_dir + '/Tile_' + str(tileNum) + '_' + bandNum + '.tif'
#         tile = IO.imread(tileLocation)
#         max = np.amax(tile)
#         print("Tile"+ str(tileNum) + "_" + bandNum + " max is " + str(max))

os.makedirs(base_dir + '/Combined_' + ClassNum)
for tileNum in myTiles:
    multiDimTiff = np.zeros(shape=(9,50,50),dtype=np.int16)
    for idx, val in enumerate(Bands):
        tileLocation = base_dir + '/Tile_' + str(tileNum) + '_' + val + '.tif'
        tile = IO.imread(tileLocation)
        multiDimTiff[idx,:,:] = tile[:,:,0]
    saveLocation = base_dir + '/Combined_' + ClassNum + '/' + 'Tile_' + str(tileNum) + '.tif'
    #saveLocation = 'Final_Tile_' + str(tileNum) + '.tif'
    tifffile.imsave(saveLocation,multiDimTiff)
#
# multiDimTiff = np.zeros(shape=(9,50,50),dtype=np.int16)
#
# tileNum = myTiles[0]
#
# for idx, val in enumerate(Bands):
#     tileLocation = base_dir + '/Tile_' + str(tileNum) + '_' + val + '.tif'
#     tile = IO.imread(tileLocation)
#     multiDimTiff[idx,:,:] = tile[:,:,0]
