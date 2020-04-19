# author: David Hurwitz
# started: 3/14/19
#

from NN_raw_data import AllLines
from NN_readRawDataFiles import ReadFilesInList
from NN_connections import ProteinConnections
from NN_formatted_data import OneNNData, NUM_PIXELS_1D
from NN_prepare_batch import OneBatch
from NN_misc import CalcAvgDistUnMatched, PIXEL_LEN, CalcBondDistStats, CalcRadiusOfGyration, AddBondDistNoise
from keras.models import Model, load_model
from keras.layers import Conv3D, BatchNormalization, Input
from keras import optimizers
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import mean_squared_error
import numpy as np
from numpy import float32

TimeStepInterval = 2
ModelNum = 12
TrainingNum = 16
InModel =  "C:/Users/david/Documents/newff/results/NN/simulations/mol_05/NN_models/model_%02d/model_%02d_training_%02d.h5"  % (ModelNum, ModelNum, TrainingNum)
ValidationBatchSize = 10

InRadius = 1.0 * PIXEL_LEN
OutRadius = 1.0 * PIXEL_LEN
InRadiusSqr = InRadius * InRadius
OutRadiusSqr = OutRadius * OutRadius

#-----------------------------------------------------------------------------------------------
# a custom NN loss function.
#-----------------------------------------------------------------------------------------------
def customLoss(yTrue, yPred):
    # check dimension on yTrue and yPred
    print("yTrue.shape = ", yTrue.shape)
    print("yPred.shape = ", yPred.shape)
    return(K.mean(K.square(yPred - yTrue), axis=-1))                    # ModelNum = 1  ('mean_squared_error')
    # return(mean_squared_error(yTrue*yTrue, yPred*yPred))              # ModelNum = 2
    # return(mean_squared_error(yTrue, yPred*yPred))                    # ModelNum = 3
    # return(K.mean(K.square(yPred - yTrue) * yTrue, axis=-1))          # ModelNum = 4
    # return(K.mean(K.square(yPred - yTrue) * (yTrue + 1.0), axis=-1))  # ModelNum = 5
    #======================================  ModelNum = 7 ====================================================
    # ones = K.ones((TrainingBatchSize, yPred.shape[1], yPred.shape[2], yPred.shape[3], yPred.shape[4]))       #
    # shift = 0.01 * ones                                                                                      #
    # diff = yPred - shift                                                                                     #
    # exponent = -1000 * diff                                                                                  #
    # exponential = K.exp(exponent)                                                                            #
    # denom = ones + exponential                                                                               #
    # quotient = ones / denom                                                                                  #
    # weighting = (9/10) * (quotient + 1/9)                                                                    #
    # return(K.mean(K.square(yPred - yTrue) * weighting, axis=-1))                                             #
    #======================================  ModelNum = 7 ====================================================

#-----------------------------------------------------------
# read CSV files that have the MD simulation data
# for testing, just use the validation data.
# allLinesValidate has the raw validation data
#-----------------------------------------------------------
CSV_Path = "C:/Users/david/Documents/newff/results/NN/simulations/mol_05/csv_files/"
FileListValidate = "sim_files_030_to_039.txt"    # 005_to_009 or 030_to_039
print("reading the validation files list in: " + CSV_Path + FileListValidate)
allLinesValidate = AllLines()
ReadFilesInList(allLinesValidate, CSV_Path, FileListValidate)

#---------------------------------------------------------------------
# look at the trajectories for a few atoms.
# 50 consecutive time-steps, direction and magnitude of step.
#---------------------------------------------------------------------
molNum = 5            # must be a molecule we read. see CSV_Path
simNum = 30           # must be a simulation we read. see FileListValidate
stepNumStart = 20000  # start of 50 consecutive time steps (multiples of 2,000, up to 100,000)
atomNums = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
for i in range(len(atomNums)):
    print("atom = %d" % (atomNums[i]))
    aLine1 = allLinesValidate.GetALine(molNum, simNum, stepNumStart, atomNums[i])
    for j in range(1, 50):
        aLine2 = allLinesValidate.GetALine(molNum, simNum, stepNumStart+j, atomNums[i])
        vec =aLine2.atomPos - aLine1.atomPos
        mag = np.linalg.norm(vec)
        dir = vec / mag
        print("%d: dir: (%f, %f, %f), mag: %f" % (j, dir[0], dir[1], dir[2], mag))
        aLine1 = aLine2

#-----------------------------------------------------------
# read the connection file for this molecule
#-----------------------------------------------------------
ConnectionFile = "C:/Users/david/Documents/newff/results/NN/simulations/mol006_sim000.connections.csv"
print("reading connections file: " + ConnectionFile)
Connections = ProteinConnections(ConnectionFile)
TotalNumConnections = Connections.getTotalNumConnections()
print("check: total num connections = " + str(TotalNumConnections))

#-------------------------------------------------------------------------------
# make a OneNNData to get array sizes for the NN
#-------------------------------------------------------------------------------
data = OneNNData(SizedForNN=True)  # default InData size is larger for extra workspace
inShape = data.InData.shape
outShape = data.OutData.shape

#-------------------------------------------------------------------------------
# make a batch of formatted data for validation
#-------------------------------------------------------------------------------
validationBatch = OneBatch(ValidationBatchSize)
validationBatch.makeABatch(allLinesValidate, Connections, InRadiusSqr, OutRadiusSqr, doRotation=True,
                           batchType='validation', timeStepInterval=TimeStepInterval)
(bigInputValidationArray, bigOutputValidationArray) = validationBatch.makeBigArrays(borderDecrease=0, combineDensityMaps=True)

# load the NN model and see if it looks right
model = load_model(InModel, custom_objects={'customLoss': customLoss})
model.compile(optimizer=Adam(lr=1e-5), loss=customLoss)
print(model.summary(line_length=150))

# calculate the loss for the validation data
print("getting loss for validation data...")
Loss = model.evaluate(bigInputValidationArray, bigOutputValidationArray, batch_size=1)
print("Validation Loss = %e" %(Loss))

# get the NN output for the validation data
print("getting output for training data...")
modelOutput = model.predict(bigInputValidationArray, batch_size=1, verbose=1)

avg0_sum = avg1_sum = avg2_sum = 0
max0_sum = max1_sum = max2_sum = 0

for index in range(ValidationBatchSize):

    # get the known atom positions for the index item in the validation batch
    data = validationBatch.getOneItem(index)

    # set the output array to have the combined outputs from the atom channels
    data.SetOutData(bigOutputValidationArray[index])
    (ktypes, kpos0) = data.GetRawData(0)
    (ktypes, kpos1) = data.GetRawData(1)

    # Append 8 extra channels to OneNNData::InData
    # Need 4 for EstimateAtomPositionsFromDensityMapsSuperAccurate
    # Need 8 for EstimateAtomPositionsFromDensityMapsSuperDuperAccurate
    np_1d = NUM_PIXELS_1D
    ExtraSpace = np.zeros(shape=(np_1d, np_1d, np_1d, 8), dtype=float32)
    data.InData = np.append(data.InData, ExtraSpace, axis=3)

    # check what the validationBatch output looks like
    # data.PrintSlice(2, 50, 'A', 0)
    data.PrintHistogram(0, 'A', 20, 'Known output')

    data.CheckForAnomalies('A', 0)
    data.CheckForAnomalies('A', 1)

    # estimate the atom positions from the density maps
    # (etypes, epos0) = data.EstimateAtomPositionsFromDensityMaps(0)
    # calculate avg dist between exact and estimate
    # (avg, max) = CalcAvgDistUnMatched(kpos0, epos0)
    # print("alg0: avg dist: %f,  max dist: %f" % (avg, max))

    # estimate the atom positions from the density maps
    # (etypes, epos1) = data.EstimateAtomPositionsFromDensityMaps(1)
    # calculate avg dist between exact and estimate
    # (avg, max) = CalcAvgDistUnMatched(kpos1, epos1)
    # print("alg0: avg dist: %f,  max dist: %f" % (avg, max))

    # estimate super-accurate atom positions from the density maps
    # (etypes, epos0) = data.EstimateAtomPositionsFromDensityMapsSuperAccurate(0, OutRadiusSqr)
    # calculate avg dist between exact and estimate
    # (avg, max) = CalcAvgDistUnMatched(kpos0, epos0)
    # print("alg1: avg dist: %f,  max dist: %f" % (avg, max))

    # estimate super-accurate atom positions from the density maps
    # (etypes, epos1) = data.EstimateAtomPositionsFromDensityMapsSuperAccurate(1, OutRadiusSqr)
    # calculate avg dist between exact and estimate
    # (avg, max) = CalcAvgDistUnMatched(kpos1, epos1)
    # print("alg1: avg dist: %f,  max dist: %f" % (avg, max))

    # estimate super-duper atom positions from the density maps
    # (etypes, epos0) = data.EstimateAtomPositionsFromDensityMapsSuperDuperAccurate(0, OutRadiusSqr)
    # calculate avg dist between exact and estimate
    # (avg, max) = CalcAvgDistUnMatched(kpos0, epos0)
    # print("alg2: avg dist: %f,  max dist: %f" % (avg, max))

    # estimate super-duper atom positions from the density maps
    # (etypes, epos1) = data.EstimateAtomPositionsFromDensityMapsSuperDuperAccurate(1, OutRadiusSqr)
    # calculate avg dist between exact and estimate
    # (avg, max) = CalcAvgDistUnMatched(kpos1, epos1)
    # print("alg2: avg dist: %f,  max dist: %f" % (avg, max))

    # use the OneNNData object to examine the model output for the index item
    data.SetOutData(modelOutput[index])

    # check what the NN output looks like
    # data.PrintSlice(2, 50, 'A', 0)
    data.PrintHistogram(0, 'A', 20, 'NN output')

    # make some minor corrections to the NN output data (when Swap=True).
    # data.CheckForAnomalies('A', 0, Swap=False)

    # estimate the atom positions from the density maps
    (etypes, epos) = data.EstimateAtomPositionsFromDensityMaps(0)
    # calculate avg dist between exact and estimate
    (avg0, max0) = CalcAvgDistUnMatched(kpos0, epos)
    print("alg0: avg dist: %f,  max dist: %f" % (avg0, max0))

    # estimate super-accurate atom positions from the density maps
    (etypes, epos) = data.EstimateAtomPositionsFromDensityMapsSuperAccurate(0, OutRadiusSqr)
    # calculate avg dist between exact and estimate
    (avg1, max1) = CalcAvgDistUnMatched(kpos0, epos)
    print("alg1: avg dist: %f,  max dist: %f" % (avg1, max1))

    # estimate super-duper atom positions from the density maps
    (etypes, epos) = data.EstimateAtomPositionsFromDensityMapsSuperDuperAccurate(0, OutRadiusSqr)
    # calculate avg dist between exact and estimate
    (avg2, max2) = CalcAvgDistUnMatched(kpos0, epos)
    print("alg2: avg dist: %f,  max dist: %f" % (avg2, max2))

    avg0_sum += avg0
    avg1_sum += avg1
    avg2_sum += avg2
    max0_sum += max0
    max1_sum += max1
    max2_sum += max2

    (min1, avg1, max1) = CalcBondDistStats(kpos0, Connections)
    RG1 = CalcRadiusOfGyration(kpos0)
    kpos0 = AddBondDistNoise(kpos0, Connections, 0.995, 0.995)
    (min2, avg2, max2) = CalcBondDistStats(kpos0, Connections)
    RG2 = CalcRadiusOfGyration(kpos0)
    kpos0 = AddBondDistNoise(kpos0, Connections, 0.995, 0.995)
    (min3, avg3, max3) = CalcBondDistStats(kpos0, Connections)
    RG3 = CalcRadiusOfGyration(kpos0)
    kpos0 = AddBondDistNoise(kpos0, Connections, 0.995, 0.995)
    (min4, avg4, max4) = CalcBondDistStats(kpos0, Connections)
    RG4 = CalcRadiusOfGyration(kpos0)
    kpos0 = AddBondDistNoise(kpos0, Connections, 0.995, 0.995)
    (min5, avg5, max5) = CalcBondDistStats(kpos0, Connections)
    RG5 = CalcRadiusOfGyration(kpos0)

    print("min1, avg1, max1: %f, %f, %f" % (min1, avg1, max1))
    print("min2, avg2, max2: %f, %f, %f" % (min2, avg2, max2))
    print("min3, avg3, max3: %f, %f, %f" % (min3, avg3, max3))
    print("min4, avg4, max4: %f, %f, %f" % (min4, avg4, max4))
    print("min5, avg5, max5: %f, %f, %f" % (min5, avg5, max5))

    print("radius of gyration 1: %f" % (RG1))
    print("radius of gyration 2: %f" % (RG2))
    print("radius of gyration 3: %f" % (RG3))
    print("radius of gyration 4: %f" % (RG4))
    print("radius of gyration 5: %f" % (RG5))


batchSize = float(ValidationBatchSize)
print("alg0_avg: avg_dist: %f,  max_dist: %f" % (avg0_sum/batchSize, max0_sum/batchSize))
print("alg1_avg: avg_dist: %f,  max_dist: %f" % (avg1_sum/batchSize, max1_sum/batchSize))
print("alg2_avg: avg_dist: %f,  max_dist: %f" % (avg2_sum/batchSize, max2_sum/batchSize))
