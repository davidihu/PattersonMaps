# author: David Hurwitz
# started: 1/25/19
#

from NN_raw_data import AllLines
from NN_readRawDataFiles import ReadFilesInList
from NN_connections import ProteinConnections
from NN_formatted_data import OneNNData, NUM_PIXELS_1D, BOX_MIN, BOX_MAX
from NN_prepare_batch import OneBatch
from NN_misc import CalcAvgDistUnMatched, PIXEL_LEN, printAtomPositions, getClosestPoints, printMatchingPositions
from NN_misc import PrintHistogram
from NN_color_scales import get_color
from keras.models import Model, load_model
from keras.layers import Conv3D, BatchNormalization, Input, Subtract, Add, Activation, Concatenate
from keras.layers import MaxPooling3D, UpSampling3D, Dense, Reshape, Flatten
from keras import optimizers
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import mean_squared_error
import numpy as np

TotalNumEpochs = 1000
NumEpochsPerWriteToFile = 1
NumEpochsPerMakeNewData = 3
NumEpochsPerViewOutput = 10
TrainingBatchSize = 3000
ValidationBatchSize = 100
NumSimulatedAnnealingSteps = 4000
HistogramStep = 0.05
UseOnlyCAlphas = False
EveryNthCAlpha = 2
MinNumAtomsPerMolecule = 10       # a positive number overrides using data from file and uses random coordinates instead
MaxNumAtomsPerMolecule = 10
NumAtomsPerMolecule = 10          # fixed at 10 for now
NumResiduesPerMolecule = 0        # a positive number overrides using all residues of a protein
AddSymmetryAtoms = True
MidSlice = round(NUM_PIXELS_1D/2)
QuarterSlice = round(NUM_PIXELS_1D/4)
ThreeQuartersSlice = round(3*NUM_PIXELS_1D/4)

AtomRadiusSqr = PIXEL_LEN * PIXEL_LEN
LR = 3e-5

# use InTrainingNum = 0 if starting from scratch
InModelNum = 8
InTrainingNum = 21

# testing
color = get_color(0.1, True, False, False, False)
color = get_color(0.3, False, False, True, False)
color = get_color(0.5, False, False, False, True)

inModel =  "C:/Users/david/Documents/newff/results/NN/Patterson_maps/NN_models/model_%02d/model_%02d_training_%02d.h5"  % (InModelNum, InModelNum, InTrainingNum)

#-----------------------------------------------------------------------------------------------
# a custom NN loss function.
#-----------------------------------------------------------------------------------------------
def customLoss(yTrue, yPred):
    # check dimension on yTrue and yPred
    print("yTrue.shape = ", yTrue.shape)
    print("yPred.shape = ", yPred.shape)
    return(K.mean(K.square(yPred - yTrue), axis=-1))                    # ModelNum = 1  ('mean_squared_error')

CSV_Path = "C:/Users/david/Documents/newff/results/NN/simulations/mol_05/csv_files/"
#-----------------------------------------------------------
# read the CSV files that have the MD simulation data
# allLinesTrain has the raw training data
# allLinesValidate has the raw validation data
#-----------------------------------------------------------
# FileListTrain = "sim_files_000_to_029.txt"       # 000_to_004 or 000_to_029
# FileListValidate = "sim_files_030_to_034.txt"    # 005_to_009 or 030_to_034
FileListTrain = "sim_files_000_to_000.txt"       # 000_to_004 or 000_to_029
FileListValidate = "sim_files_000_to_000.txt"    # 005_to_009 or 030_to_034
print("reading the training files listed in: " + CSV_Path + FileListTrain)
allLinesTrain = AllLines()
ReadFilesInList(allLinesTrain, CSV_Path, FileListTrain)
print("reading the validation files list in: " + CSV_Path + FileListValidate)
allLinesValidate = AllLines()
ReadFilesInList(allLinesValidate, CSV_Path, FileListValidate)

#-------------------------------------------------------------------------------
# make a OneNNData to get array sizes for the NN
#-------------------------------------------------------------------------------
data = OneNNData()
inShape = data.InData.shape
outShape = data.OutData.shape

#-------------------------------------------------------------------------------
# make the Keras Functional API model.
#-------------------------------------------------------------------------------
input =   Input(shape=data.InData.shape)
L01a =    Conv3D(20, 5, activation='relu', padding='same', kernel_initializer='he_normal')  (input)
L01b =    Conv3D(20, 5, activation='relu', padding='same', kernel_initializer='he_normal')  (L01a)
D02 =     MaxPooling3D(pool_size=(2,2,2))                                                   (L01b)
L08a =    Conv3D(20, 7, activation='relu', padding='same', kernel_initializer='he_normal')  (D02)
L08b =    Conv3D(20, 7, activation='relu', padding='same', kernel_initializer='he_normal')  (L08a)
L08c =    Conv3D(20, 7, activation='relu', padding='same', kernel_initializer='he_normal')  (L08b)
L08d =    Conv3D(20, 7, activation='relu', padding='same', kernel_initializer='he_normal')  (L08c)
L08e =    Conv3D(20, 7, activation='relu', padding='same', kernel_initializer='he_normal')  (L08d)
L08f =    Conv3D(20, 7, activation='relu', padding='same', kernel_initializer='he_normal')  (L08e)
L08g =    Conv3D(20, 7, activation='relu', padding='same', kernel_initializer='he_normal')  (L08f)
L08h =    Conv3D(20, 7, activation='relu', padding='same', kernel_initializer='he_normal')  (L08g)
U03 =     UpSampling3D(size=(2,2,2))                                                        (L08h)
L15a =    Conv3D(20, 5, activation='relu', padding='same', kernel_initializer='he_normal')  (U03)
output =  Conv3D( 1, 5, activation='tanh', padding='same', kernel_initializer='he_normal')  (L15a)
model =   Model(inputs=[input], outputs=[output])
print(model.summary(line_length=150))
model.compile(optimizer=Adam(lr=LR), loss=customLoss)

#-------------------------------------------------------------------------------
# make a batch of formatted data for validation
#-------------------------------------------------------------------------------
validationBatch = OneBatch(ValidationBatchSize)

# make the validation batch
validationBatch.makeABatch(allLinesValidate, AtomRadiusSqr, doRotation=True, batchType='validation',
                           OnlyCAlphas=UseOnlyCAlphas, NthCAlpha=EveryNthCAlpha,
                           MinNumAtoms=MinNumAtomsPerMolecule, MaxNumAtoms=MaxNumAtomsPerMolecule,
                           NumResidues=NumResiduesPerMolecule, AddSymmetryAtoms=AddSymmetryAtoms)
(bigInputValidationArray, bigOutputValidationArray) = validationBatch.makeBigArrays()

# load and compile the NN model
if InTrainingNum >= 1:
    model = load_model(inModel, custom_objects={'customLoss': customLoss})
    model.compile(optimizer=Adam(lr=LR), loss=customLoss)

# get the NN output for the validation data.
print("getting output for validation data...")
modelOutput = model.predict(bigInputValidationArray, batch_size=1, verbose=2)

# calculate the loss for the validation data
print("getting loss for validation data...")
Loss = model.evaluate(bigInputValidationArray, bigOutputValidationArray, batch_size=1, verbose=2)
print("Validation Loss = %e" % (Loss))

# distances between true and estimated positions for multiple tests
allDistances = []

# for each NN output map
for i in range(ValidationBatchSize):

    # check the true output
    data.OutData = bigOutputValidationArray[i]
    data.PrintSlice(2, MidSlice, False, 0)
    data.PrintHistogram(False, 20, 'true output', 0)
    # data.Print3d(False)

    # create a density map on the NN output from saved atom positions
    # add density for the centro-symmetrically related positions
    # draw the density map in 3d, coloring the 2 sets of atoms differently
    data = validationBatch.getOneItem(i)
    # data.Print2d(MidSlice,   "True")
    # data.Print2dProjection(0, "True")
    # data.Print2dProjection(1, "True")
    # data.Print2dProjection(2, "True")
    data.Print3dTest3(AtomRadiusSqr, 0.05, "original atoms",                    i, True, False, Opacity=1.0)
    data.Print3dTest3(AtomRadiusSqr, 0.05, "centro-symmetric atoms",            i, False, True, Opacity=1.0)
    data.Print3dTest3(AtomRadiusSqr, 0.05, "original + centro-symmetric atoms", i, True,  True, Opacity=1.0)

    # check the NN output
    data.OutData = modelOutput[i]
    data.PrintSlice(2, MidSlice, False, 0)
    data.PrintHistogram(False, 20, 'NN output', 0)

    # draw the density map predicted by the NN in 3d
    # data.Print2d(MidSlice,   "Predicted")
    # data.Print2dProjection(0, "Predicted")
    # data.Print2dProjection(1, "Predicted")
    # data.Print2dProjection(2, "Predicted")
    data.Print3dTest2(0.05, "predicted atoms", i, Opacity=1.0)

    # check the NN input
    # data.InData = bigInputValidationArray[i]
    # data.PrintSlice(2, MidSlice, True, 0)
    # data.PrintHistogram(True, 20, 'true input', 0)

    # get pixels that are also local maxima
    (Pixels1, PixelVals1, PixelSums1) = data.GetPeaks1(0.25)

    # estimate atom positions from these pixels
    estimatedPositions = data.EstimateAtomPositions(Pixels1)

    # get pixels-sums that are also local maxima
    (Pixels2, PixelVals2, PixelSums2) = data.GetPeaks4(0.25, Pixels1)

    # estimate atom positions from these pixels
    estimatedPositions2 = data.EstimateAtomPositions(Pixels2)

    # combine the 2 sets of atom-position estimates
    if len(Pixels2) > 0:
        estimatedPositions = np.append(estimatedPositions, estimatedPositions2, axis=0)

    # combine the atom-position estimates with their centrosymmetric counterparts
    estimatedPositions = np.append(estimatedPositions, estimatedPositions*-1.0, axis=0)

    # get the actual atom positions
    data = validationBatch.getOneItem(i)
    atomPositions = np.array(data.AtomPositions)

    # for each atomPosition, find the estimatedPosition its closest to
    # (closestIndices, closestDistances, avgDist) = getClosestPoints(atomPositions, estimatedPositions)
    # printAtomPositions(atomPositions, "true atom positions:", closestIndices, closestDistances)

    # print the true atom positions
    printAtomPositions(atomPositions, "true atom positions:", None, None)

    # print the estimated atom positions
    printAtomPositions(estimatedPositions, "estimated atom positions:", None, None)

    # get a set of atoms that gives the closest match to the known Patterson map
    # data.InData = bigInputValidationArray[i]
    subset = data.PickBestSet(estimatedPositions, NumAtomsPerMolecule, AtomRadiusSqr, NumSimulatedAnnealingSteps)
    # printAtomPositions(subset, "selected atom positions:", None, None)

    # print the true atom positions alongside their matching estimated positions
    distances = printMatchingPositions(atomPositions, subset)
    allDistances += distances

    # periodically print atom accuracy histogram
    if i%10 == 0:
        print("atom-accuracy histogram after %d cases:" % (i+1))
        PrintHistogram(allDistances, 0.0, 6.0, HistogramStep)

PrintHistogram(allDistances, 0.0, 6.0, HistogramStep)
test = 1
