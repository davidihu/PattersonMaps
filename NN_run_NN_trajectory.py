# author: David Hurwitz
# started: 3/14/19
#

from NN_raw_data import AllLines
from NN_readRawDataFiles import ReadFilesInList
from NN_connections import ProteinConnections
from NN_formatted_data import OneNNData, NUM_PIXELS_1D
from NN_prepare_batch import OneBatch
from NN_misc import CalcAvgDistUnMatched, CalcAvgDist, PIXEL_LEN, MatchVectorOrder, MakeSimilarPDB
from NN_misc import CalcRadiusOfGyration, CalcBondDistStats, CalcNearestNonBonded
from keras.models import Model, load_model
from keras.layers import Conv3D, BatchNormalization, Input
from keras import optimizers
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import mean_squared_error
import numpy as np
from numpy import float32

BatchSize = 1

TimeStepInterval = 2
ModelNum = 12
TrainingNum = 16
InModel =  "C:/Users/david/Documents/newff/results/NN/simulations/mol_05/NN_models/model_%02d/model_%02d_training_%02d.h5"  % (ModelNum, ModelNum, TrainingNum)

InRadius = 1.0 * PIXEL_LEN
OutRadius = 1.0 * PIXEL_LEN
InRadiusSqr = InRadius * InRadius
OutRadiusSqr = OutRadius * OutRadius

#-----------------------------------------------------------------------------------------------
# custom NN loss function.
#-----------------------------------------------------------------------------------------------
def customLoss(yTrue, yPred):
    # check dimension on yTrue and yPred
    print("yTrue.shape = ", yTrue.shape)
    print("yPred.shape = ", yPred.shape)
    return(K.mean(K.square(yPred - yTrue), axis=-1))                    # ModelNum = 1  ('mean_squared_error')

#-----------------------------------------------------------
# read CSV files that have the MD simulation data.
# for testing, use the validation data.
# allLinesValidate has the raw validation data.
#-----------------------------------------------------------
CSV_Path = "C:/Users/david/Documents/newff/results/NN/simulations/mol_05/csv_files/"
FileListValidate = "sim_files_030_to_039.txt"    # 005_to_009 or 030_to_039
print("reading the validation files list in: " + CSV_Path + FileListValidate)
allLinesValidate = AllLines()
ReadFilesInList(allLinesValidate, CSV_Path, FileListValidate)

#-----------------------------------------------------------
# read the connection file for this molecule
#-----------------------------------------------------------
ConnectionFile = "C:/Users/david/Documents/newff/results/NN/simulations/mol006_sim000.connections.csv"
print("reading connections file: " + ConnectionFile)
Connections = ProteinConnections(ConnectionFile)
TotalNumConnections = Connections.getTotalNumConnections()
print("check: total num connections = " + str(TotalNumConnections))

#-------------------------------------------------------------------------------
# make a OneNNData for general use.
#-------------------------------------------------------------------------------
data = OneNNData(SizedForNN=True)

#-------------------------------------------------------------------------------
# Append 8 extra channels to OneNNData::InData
# Need 4 for EstimateAtomPositionsFromDensityMapsSuperAccurate
# Need 8 for EstimateAtomPositionsFromDensityMapsSuperDuperAccurate
#-------------------------------------------------------------------------------
np_1d = NUM_PIXELS_1D
ExtraSpace = np.zeros(shape=(np_1d, np_1d, np_1d, 8), dtype=float32)
data.InData = np.append(data.InData, ExtraSpace, axis=3)

#-------------------------------------------------------------------------------
# make the first NN input for the trajectory
#-------------------------------------------------------------------------------
Batch = OneBatch(BatchSize)
Batch.makeABatch(allLinesValidate, Connections, InRadiusSqr, OutRadiusSqr, doRotation=True, batchType='validation',
                 flipTimeStepOrder=True, timeStepInterval=TimeStepInterval)
(InputArray, OutputArray) = Batch.makeBigArrays(borderDecrease=0, combineDensityMaps=True)

# load the NN model and see if it looks right
model = load_model(InModel, custom_objects={'customLoss': customLoss})
model.compile(optimizer=Adam(lr=1e-5), loss=customLoss)
print(model.summary(line_length=150))

# calculate the loss for the validation data
print("getting loss for validation data...")
Loss = model.evaluate(InputArray, OutputArray, batch_size=1)
print("Validation Loss = %e" %(Loss))

# set "data" to the 1st batch data. this includes raw atom coordinates.
data = Batch.getOneItem(0)

# as long as the atom coordinates in the same order, can use this template for making pdbs
template = "C:/Users/david/Documents/newff/results/NN/simulations/mol_05/pdb_files/tmp200600.pdb"

# repeat many times
for i in range(100):

    # get the NN output
    print("getting output for training data...")
    modelOutput = model.predict(InputArray, batch_size=1, verbose=1)

    # calculate atom positions from the NN output. these are the time-step-0 positions.
    data.SetOutData(modelOutput[0])
    data.PrintHistogram(0, 'A', 20, 'NN output')
    # data.PrintSlice(2, 50, 'A', 4)
    # data.PrintSlice(2, 50, 'A', 3)
    # data.PrintSlice(2, 50, 'A', 2)
    # data.PrintSlice(2, 50, 'A', 1)
    data.PrintSlice(2, 50, 'A', 0)
    (types, timestep0_pos) = data.EstimateAtomPositionsFromDensityMapsSuperAccurate(0, InRadiusSqr)

    # get the current time-step-1 positions. put time-step-0 positions in the same order.
    timestep1_pos = data.AllAtomPositions[1]
    timestep0_pos = MatchVectorOrder(timestep1_pos, timestep0_pos)

    # shift the formatted data by one time-step
    data.ShiftDataOneTimeStep()

    # shift time-step-0 positions to time-step-1
    data.AllAtomPositions[1] = timestep0_pos

    # make PDB file for coordinates calculated from NN output
    outPDB = "%d.pdb" % (i)
    MakeSimilarPDB(template, outPDB, data.AllAtomPositions[1])

    # make density maps for time-step-1 from the atom positions that were just shifted in.
    data.ClearDensityMapsForOneTimeStep(1)
    data.MakeMaps(data.AllAtomPositions[1], Connections, types, 1, InRadiusSqr)

    # put data with the shifted time step into Batch
    Batch.setOneItem(0, data)

    # make the new InputArray for the NN
    (InputArray, OutputArray) = Batch.makeBigArrays(borderDecrease=0, combineDensityMaps=True)

    # calculate the average distance the atoms move between time steps
    (avg43, max43) = CalcAvgDist(data.AllAtomPositions[4], data.AllAtomPositions[3])
    (avg32, max32) = CalcAvgDist(data.AllAtomPositions[3], data.AllAtomPositions[2])
    (avg21, max21) = CalcAvgDist(data.AllAtomPositions[2], data.AllAtomPositions[1])

    print("average atom movements between time-steps (avg, max)")
    print("4 -> 3: (%f, %f)" % (avg43, max43))
    print("3 -> 2: (%f, %f)" % (avg32, max32))
    print("2 -> 1: (%f, %f)" % (avg21, max21))

    RG4 = CalcRadiusOfGyration(data.AllAtomPositions[4])
    RG3 = CalcRadiusOfGyration(data.AllAtomPositions[3])
    RG2 = CalcRadiusOfGyration(data.AllAtomPositions[2])
    RG1 = CalcRadiusOfGyration(data.AllAtomPositions[1])

    print("radius of gyration 4: %f" % (RG4))
    print("radius of gyration 3: %f" % (RG3))
    print("radius of gyration 2: %f" % (RG2))
    print("radius of gyration 1: %f" % (RG1))

    (min4, avg4, max4) = CalcBondDistStats(data.AllAtomPositions[4], Connections)
    (min3, avg3, max3) = CalcBondDistStats(data.AllAtomPositions[3], Connections)
    (min2, avg2, max2) = CalcBondDistStats(data.AllAtomPositions[2], Connections)
    (min1, avg1, max1) = CalcBondDistStats(data.AllAtomPositions[1], Connections)

    print("min4, avg4, max4: %f, %f, %f" % (min4, avg4, max4))
    print("min3, avg3, max3: %f, %f, %f" % (min3, avg3, max3))
    print("min2, avg2, max2: %f, %f, %f" % (min2, avg2, max2))
    print("min1, avg1, max1: %f, %f, %f" % (min1, avg1, max1))

    minNonBonded4 = CalcNearestNonBonded(data.AllAtomPositions[4], Connections)
    minNonBonded3 = CalcNearestNonBonded(data.AllAtomPositions[3], Connections)
    minNonBonded2 = CalcNearestNonBonded(data.AllAtomPositions[2], Connections)
    minNonBonded1 = CalcNearestNonBonded(data.AllAtomPositions[1], Connections)

    print("closest non-bonded 4: %f" % (minNonBonded4))
    print("closest non-bonded 3: %f" % (minNonBonded3))
    print("closest non-bonded 2: %f" % (minNonBonded2))
    print("closest non-bonded 1: %f" % (minNonBonded1))
