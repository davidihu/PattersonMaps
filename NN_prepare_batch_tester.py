# author: David Hurwitz
# started: 3/28/18
#

from NN_raw_data import ALine
from NN_connections import ProteinConnections
from NN_raw_data import AllLines
from NN_formatted_data import OneNNData, NUM_PIXELS_1D
from NN_readRawDataFiles import ReadFilesInList
from NN_prepare_batch import OneBatch
from NN_misc import CalcRMSD, MakePDB, MoveToCenter, CalcAvgDistUnMatched, CalcAvgDist, NUM_1D, PIXEL_LEN
import numpy as np
from numpy import float32

AtomRadius = 1.0 * PIXEL_LEN
AtomRadiusSqr = AtomRadius * AtomRadius

CSV_Path = "C:/Users/david/Documents/newff/results/NN/simulations/mol_05/csv_files/"
FileList = "sim_files_005_to_009.txt"

print("reading the files listed in: " + CSV_Path + FileList)

# read in all the CSV files that hold the MD simulation data
allLines = AllLines()
ReadFilesInList(allLines, CSV_Path, FileList)

print("num lines = " + str(allLines.getNumLines()))
print("num molecules = " + str(allLines.GetNumMolecules()))
print("num simulations per molecule = " + str(allLines.GetNumSimulations(allLines.GetMolNumStart())))
print("num time steps per simulation = " + str(allLines.GetNumTimeSteps(allLines.GetMolNumStart(), 5)) + " (many are skipped)")

# for the record
print("NUM_1D = %d" % (NUM_1D))

# make a batch of NN input and output data
batchData = OneBatch(10)   # 10 or 100 or ...
batchData.makeABatch(allLines, AtomRadiusSqr, doRotation=True, batchType='training')

# get the batch arrays into a format that is suitable for the NN: [batchsize, nx, ny, nz, numChannels]
(bigInputArray, bigOutputArray) = batchData.makeBigArrays()
print("bigInputArray.shape = ", bigInputArray.shape)
print("bigOutputArray.shape = ", bigOutputArray.shape)

# check that the shape is right for the NN
OneInput = bigInputArray[0]
OneOutput = bigOutputArray[0]
print("OneInput.shape = ", OneInput.shape)
print("OneOutput.shape = ", OneOutput.shape)

# leaving off here

# create this OneNNData object so we can use its member functions
data = OneNNData(SizedForNN=True)

# look at one slice of one output density map at different atom-radii
batchData.getOneItem(0).OutData = bigOutputArray[0]
batchData.getOneItem(0).PrintSlice(2, 50, False)
batchData.getOneItem(0).PrintHistogram(False, 20, 'test data')

batchData.getOneItem(0).ReMakeMap(False, 1.50*PIXEL_LEN)
(bigInputArray2, bigOutputArray2) = batchData.makeBigArrays()
batchData.getOneItem(0).OutData = bigOutputArray[0]
batchData.getOneItem(0).PrintSlice(2, 50, False)
batchData.getOneItem(0).PrintHistogram(False, 20, 'test data')

batchData.getOneItem(0).ReMakeMap(False, 2.00*PIXEL_LEN)
(bigInputArray3, bigOutputArray3) = batchData.makeBigArrays()
batchData.getOneItem(0).OutData = bigOutputArray[0]
batchData.getOneItem(0).PrintSlice(2, 50, False)
batchData.getOneItem(0).PrintHistogram(False, 20, 'test data')

# for each training example in batchData, look at the histograms
for sampleIndex in range(batchData.getNumInBatch()):
    data.OutData = bigOutputArray2[sampleIndex]
    data.PrintHistogram(0, 'A', 20, '')

# for determining the avg distance between atom positions and estimated atom positions
sum_of_avgs1 = 0
sum_of_avgs2 = 0
num_avgs = 0


avg_dist1 = sum_of_avgs1 / num_avgs
avg_dist2 = sum_of_avgs2 / num_avgs

print("avg distance between raw and estimated atom positions = " + str(avg_dist1))
print("avg distance between raw and super-accurate estimated atom positions = " + str(avg_dist2))

# memory checks:  allocate arrays for the NN batch
batchData = OneBatch(100)
# repeat a few times
for batchIndex in range(1):
    # make one batch of training examples (randomly chosen conformations from the simulations)
    batchData.makeABatch(allLines, AtomRadiusSqr, doRotation=True)
    print("%3d: made a batch of size %4d training examples" %(batchIndex+1, batchData.getNumInBatch()))
    # get stacked arrays to present to the NN
    (bigInputArray, bigOutputArray) = batchData.makeBigArrays()

test = 1
