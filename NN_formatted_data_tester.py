# author: David Hurwitz
# started: 3/14/18
#
# tests of NN_formatted_data.py

import numpy as np
from NN_readRawDataFiles    import ReadFilesInList
from NN_connections         import ProteinConnections
from NN_raw_data            import AllLines
from NN_formatted_data      import OneNNData, BOX_MIN, BOX_MAX, NUM_PIXELS_1D
from NN_misc                import CalcRMSD, MakePDB, MoveToCenter, Get3RandomAngles, RotateProtein
from NN_misc                import RES_TYPES, STEP_NUM_INC
from NN_misc                import SMALL_NUM, PIXEL_LEN, CalcPixelOccupancy, CalcPixelOccupancyNoRecursion
from NN_misc                import CalcPixelOccupancyUsingRandomNumbers, CalcAvgDist, CalcAvgDistUnMatched
from numpy import float32

AtomRadius = 1.0 * PIXEL_LEN
AtomRadiusSqr = AtomRadius * AtomRadius
AtomRadiusCubed = AtomRadius * AtomRadius * AtomRadius

molNum = 5
CSV_Path = "C:/Users/david/Documents/newff/results/NN/simulations/mol_05/csv_files/"
FileList = "sim_files_000_to_000.txt"      # 1 simulation file is in this list

# read in all the CSV files that hold the MD simulation data
# the data is stored in an AllLines object
Lines = AllLines()
ReadFilesInList(Lines, CSV_Path, FileList)

data = OneNNData()  # make empty data arrays

#---------------------------------
# test (x, y, z) -> (i, j, k)
#---------------------------------
sn = SMALL_NUM
ps2 = PIXEL_LEN / 2
ps9 = 9*PIXEL_LEN / 10
np1 = NUM_PIXELS_1D - 1
np2 = int(NUM_PIXELS_1D / 2)
np21 = int(NUM_PIXELS_1D / 2) - 1

# test: (BOX_MIN+sn, BOX_MIN+sn, BOX_MIN+sn) -> (0,0,0)
Pos = (BOX_MIN+sn, BOX_MIN+sn, BOX_MIN+sn)
(i,j,k) = data.GetPixel(Pos)
assert(i == 0 and j == 0 and k == 0)

# test: (BOX_MIN+0.5*pixel, BOX_MIN+0.5*pixel, BOX_MIN+0.5*pixel) -> (0,0,0)
Pos = (BOX_MIN+ps2, BOX_MIN+ps2, BOX_MIN+ps2)
(i,j,k) = data.GetPixel(Pos)
assert(i == 0 and j == 0 and k == 0)

# test: (BOX_MIN+0.9*pixel, BOX_MIN+0.9*pixel, BOX_MIN+0.9*pixel) -> (0,0,0)
Pos = (BOX_MIN+ps9, BOX_MIN+ps9, BOX_MIN+ps9)
(i,j,k) = data.GetPixel(Pos)
assert(i == 0 and j == 0 and k == 0)

# test: (BOX_MAX-sn, BOX_MAX-sn, BOX_MAX-sn) -> (NUM_PIXELS-1,NUM_PIXELS-1,NUM_PIXELS-1)
Pos = (BOX_MAX-sn, BOX_MAX-sn, BOX_MAX-sn)
(i,j,k) = data.GetPixel(Pos)
assert(i == np1 and j == np1 and k == np1)

# test: (BOX_MAX-0.5*pixel, BOX_MAX-0.5*pixel, BOX_MAX-0.5*pixel) -> (NUM_PIXELS-1,NUM_PIXELS-1,NUM_PIXELS-1)
Pos = (BOX_MAX-ps2, BOX_MAX-ps2, BOX_MAX-ps2)
(i,j,k) = data.GetPixel(Pos)
assert(i == np1 and j == np1 and k == np1)

# test: (BOX_MAX-0.9*pixel, BOX_MAX-0.9*pixel, BOX_MAX-0.9*pixel) -> (NUM_PIXELS-1,NUM_PIXELS-1,NUM_PIXELS-1)
Pos = (BOX_MAX-ps9, BOX_MAX-ps9, BOX_MAX-ps9)
(i,j,k) = data.GetPixel(Pos)
assert(i == np1 and j == np1 and k == np1)

# test: (0+sn, 0+sn, 0+sn) -> (NUM_PIXELS/2,   NUM_PIXELS/2,   NUM_PIXELS/2
Pos = (0+sn,0+sn,0+sn)
(i,j,k) = data.GetPixel(Pos)
assert(i == np2 and j == np2 and k == np2)

# test: (0-sn, 0-sn, 0-sn) -> (NUM_PIXELS/2-1, NUM_PIXELS/2-1, NUM_PIXELS/2-1
Pos = (0-sn,0-sn,0-sn)
(i,j,k) = data.GetPixel(Pos)
assert(i == np21 and j == np21 and k == np21)

# test DoesPixelHaveSomeOccupancy from a pixel corner to a neighboring pixel
Pt = (sn, sn, sn)
Pixel = data.GetPixel(Pt)
TestPixel = (Pixel[0]+1, Pixel[1]+1, Pixel[2]+0)
hasOccupancy = data.DoesPixelHaveSomeOccupancy(Pt, TestPixel, AtomRadiusSqr)
assert(hasOccupancy == False)

# test DoesPixelHaveSomeOccupancy from a pixel corner to a neighboring pixel
Pt = (sn, sn, sn)
Pixel = data.GetPixel(Pt)
TestPixel = (Pixel[0]-1, Pixel[1]-1, Pixel[2]+0)
hasOccupancy = data.DoesPixelHaveSomeOccupancy(Pt, TestPixel, AtomRadiusSqr)
assert(hasOccupancy == True)

# test DoesPixelHaveSomeOccupancy from a pixel center to a neighboring pixel
Pt = (ps2, ps2, ps2)
Pixel = data.GetPixel(Pt)
TestPixel = (Pixel[0]+1, Pixel[1]+1, Pixel[2]+0)
hasOccupancy = data.DoesPixelHaveSomeOccupancy(Pt, TestPixel, AtomRadiusSqr)
assert(hasOccupancy == True)

Pos = data.Get000Pos((25, 17, 52))
Pos = (Pos[0]+sn, Pos[1]+PIXEL_LEN-sn, Pos[2]+PIXEL_LEN/2)

# test DoesPixelHaveSomeOccupancy.
# of the 27 pixels below (Pos + its neighbors), 16 should have some occupancy.
# for the exact matches, see p. 109 of my 2018 notebook.
count = 0
for i in range(-1,2):
    for j in range(-1,2):
        for k in range(-1,2):
            TestPixel = (25+i, 17+j, 52+k)
            if data.DoesPixelHaveSomeOccupancy(Pos, TestPixel, AtomRadiusSqr):
                count = count+1
assert(count == 16)

# test CalcPixelOccupancy
AtomRadius = PIXEL_LEN
AtomRadiusSqr = AtomRadius * AtomRadius
NumCubes = [0]
fraction2 = CalcPixelOccupancy((0.25, 0.25, 0.25), (0.0, 0.0, 0.5), 2, NumCubes, AtomRadiusSqr)
fraction3 = CalcPixelOccupancy((0.25, 0.25, 0.25), (0.0, 0.0, 0.5), 3, NumCubes, AtomRadiusSqr)
fraction4 = CalcPixelOccupancy((0.25, 0.25, 0.25), (0.0, 0.0, 0.5), 4, NumCubes, AtomRadiusSqr)
fraction5 = CalcPixelOccupancy((0.25, 0.25, 0.25), (0.0, 0.0, 0.5), 5, NumCubes, AtomRadiusSqr)
fraction6 = CalcPixelOccupancy((0.25, 0.25, 0.25), (0.0, 0.0, 0.5), 6, NumCubes, AtomRadiusSqr)

fract1 = CalcPixelOccupancy((0,0,0), (0,           0,           0          ), 3, NumCubes, AtomRadiusSqr)
fract2 = CalcPixelOccupancy((0,0,0), (0,           0,           0-PIXEL_LEN), 4, NumCubes, AtomRadiusSqr)
fract3 = CalcPixelOccupancy((0,0,0), (0,           0-PIXEL_LEN, 0          ), 5, NumCubes, AtomRadiusSqr)
fract4 = CalcPixelOccupancy((0,0,0), (0,           0-PIXEL_LEN, 0-PIXEL_LEN), 6, NumCubes, AtomRadiusSqr)
fract5 = CalcPixelOccupancy((0,0,0), (0-PIXEL_LEN, 0,           0          ), 3, NumCubes, AtomRadiusSqr)
fract6 = CalcPixelOccupancy((0,0,0), (0-PIXEL_LEN, 0,           0-PIXEL_LEN), 4, NumCubes, AtomRadiusSqr)
fract7 = CalcPixelOccupancy((0,0,0), (0-PIXEL_LEN, 0-PIXEL_LEN, 0          ), 5, NumCubes, AtomRadiusSqr)
fract8 = CalcPixelOccupancy((0,0,0), (0-PIXEL_LEN, 0-PIXEL_LEN, 0-PIXEL_LEN), 6, NumCubes, AtomRadiusSqr)

frac1 = CalcPixelOccupancyNoRecursion((0,0,0), (0,0,0), NumCubes, AtomRadiusSqr)
frac2 = CalcPixelOccupancyUsingRandomNumbers((0,0,0), (0,0,0),   500, AtomRadiusSqr)
frac3 = CalcPixelOccupancyUsingRandomNumbers((0,0,0), (0,0,0),  2000, AtomRadiusSqr)
frac4 = CalcPixelOccupancyUsingRandomNumbers((0,0,0), (0,0,0), 10000, AtomRadiusSqr)
frac5 = CalcPixelOccupancyUsingRandomNumbers((0,0,0), (0,0,0), 40000, AtomRadiusSqr)

# test making the 3d arrays of data
# ktypes = known atom types, kpos = known atom positions
(ktypes, kpos) = data.SetOneTrainingData(Lines, 5, 0, 2025, AtomRadiusSqr, doRotation=True)

# look at console output
data.CountSetPixels()

data.PrintHistogram(True, 20, 'input density')
data.PrintHistogram(True, 100, 'input density')
data.PrintHistogram(False, 20, 'output density')

# print adjacent Y-Z slices from the input
data.PrintSlice(0, 30, True)
data.PrintSlice(0, 31, True)
data.PrintSlice(0, 32, True)
data.PrintSlice(0, 33, True)
data.PrintSlice(0, 34, True)

# print X-Y slices from the input and output
data.PrintSlice(2, 32, True)
data.PrintSlice(2, 32, False)

# experimenting with making 3d images
data.Print3d(False)
data.Print3d(True)

test = 1
