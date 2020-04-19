# author: David Hurwitz
# started: 1/25/19
#

from NN_raw_data import AllLines
from NN_readRawDataFiles import ReadFilesInList
from NN_connections import ProteinConnections
from NN_formatted_data import OneNNData, NUM_PIXELS_1D, BOX_MIN, BOX_MAX
from NN_prepare_batch import OneBatch
from NN_misc import CalcAvgDistUnMatched, PIXEL_LEN
from keras.models import Model, load_model
from keras.layers import Conv3D, BatchNormalization, Input, Subtract, Add, Activation, Concatenate
from keras.layers import MaxPooling3D, UpSampling3D, Dense, Reshape, Flatten
from keras import optimizers
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import mean_squared_error
import numpy as np

TotalNumEpochs = 100
NumEpochsPerWriteToFile = 1
NumEpochsPerMakeNewData = 3
NumEpochsPerViewOutput = 10
TrainingBatchSize = 3000
ValidationBatchSize = 300
UseOnlyCAlphas = False
EveryNthCAlpha = 2
MinNumAtomsPerMolecule = 10       # a positive number overrides using data from file and uses random coordinates instead
MaxNumAtomsPerMolecule = 10
NumResiduesPerMolecule = 0        # a positive number overrides using all residues of a protein
AddSymmetryAtoms = True
MidSlice = round(NUM_PIXELS_1D/2)
QuarterSlice = round(NUM_PIXELS_1D/4)
ThreeQuartersSlice = round(3*NUM_PIXELS_1D/4)

AtomRadiusSqr = PIXEL_LEN * PIXEL_LEN

# mostly I used 4e-5. model/training 8/21 used 5e-6.
# so I guess I decreased this in the later trainings models 1/86-90 and 8/1-21.
LR = 4e-5

# use InTrainingNum = 0 if starting from scratch
#InModelNum = 8                   # last good start
#InTrainingNum = 20               # .
#OutModelNum = 8                  # last good stop
#OutTrainingNum = 21              # .

InModelNum = 26
InTrainingNum = 0
OutModelNum = 26
OutTrainingNum = 1

inModel =  "C:/Users/david/Documents/newff/results/NN/Patterson_maps/NN_models/model_%02d/model_%02d_training_%02d.h5"  % (InModelNum, InModelNum, InTrainingNum)
outModel = "C:/Users/david/Documents/newff/results/NN/Patterson_maps/NN_models/model_%02d/model_%02d_training_%02d.h5"  % (OutModelNum, OutModelNum, OutTrainingNum)
outTxt =   "C:/Users/david/Documents/newff/results/NN/Patterson_maps/NN_models/model_%02d/model_%02d_training_%02d.txt" % (OutModelNum, OutModelNum, OutTrainingNum)

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

CSV_Path = "C:/Users/david/Documents/newff/results/NN/simulations/mol_05/csv_files/"
#-----------------------------------------------------------
# read the CSV files that have the MD simulation data
# allLinesTrain has the raw training data
# allLinesValidate has the raw validation data
#-----------------------------------------------------------
# FileListTrain = "sim_files_000_to_029.txt"       # 000_to_004 or 000_to_029
# FileListValidate = "sim_files_030_to_034.txt"    # 005_to_009 or 030_to_034
FileListTrain = "sim_files_000_to_000.txt"       # 000_to_004 or 000_to_029
FileListValidate = "sim_files_030_to_030.txt"    # 005_to_009 or 030_to_034
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
# make a batch of formatted data for training
# make a batch of formatted data for validation
#-------------------------------------------------------------------------------
trainingBatch = OneBatch(TrainingBatchSize)
validationBatch = OneBatch(ValidationBatchSize)

# make the validation batch
validationBatch.makeABatch(allLinesValidate, AtomRadiusSqr, doRotation=True, batchType='validation',
                           OnlyCAlphas=UseOnlyCAlphas, NthCAlpha=EveryNthCAlpha,
                           MinNumAtoms=MinNumAtomsPerMolecule, MaxNumAtoms=MaxNumAtomsPerMolecule,
                           NumResidues=NumResiduesPerMolecule, AddSymmetryAtoms=AddSymmetryAtoms)
(bigInputValidationArray, bigOutputValidationArray) = validationBatch.makeBigArrays()

# it's only for debugging that I'm doing this more than once
for repeat in range(1):

    # make the training batch
    trainingBatch.makeABatch(allLinesTrain, AtomRadiusSqr, doRotation=True, batchType='training',
                             OnlyCAlphas=UseOnlyCAlphas, NthCAlpha=EveryNthCAlpha,
                             MinNumAtoms=MinNumAtomsPerMolecule, MaxNumAtoms=MaxNumAtomsPerMolecule,
                             NumResidues=NumResiduesPerMolecule, AddSymmetryAtoms=AddSymmetryAtoms,
                             printNumPixelsTouchedInOutputDensity=False)
    (bigInputTrainingArray, bigOutputTrainingArray) = trainingBatch.makeBigArrays()

    # check atom separation for some training cases
    closest = 999
    farthest = -999
    farthest_from_center = -999
    for i in range(TrainingBatchSize):
        example = trainingBatch.getOneItem(i)
        dist1 = example.GetClosestPairDist()
        dist2 = example.GetFarthestPairDist()
        dist3 = example.GetFarthestFromCenter()
        if dist1 < closest:
            closest = dist1
            print("closest closest-pair-distance = %f" % (closest))
        if dist2 > farthest:
            farthest = dist2
            print("farthest farthest-pair-distance = %f" % (farthest))
        if dist3 > farthest_from_center:
            farthest_from_center = dist3
            print("farthest farthest-from-center = %f" % (farthest_from_center))

# check a few slices of the trainingBatch input of the 1st training case (the Patterson maps)
data.InData = bigInputTrainingArray[0]
data.PrintSlice(2, MidSlice-1, True, 0)
data.PrintSlice(2, MidSlice,   True, 0)
data.PrintSlice(2, MidSlice+1, True, 0)

# check the trainingBatch input histogram for a few training cases
for i in range(TrainingBatchSize):
    data.InData = bigInputTrainingArray[i]
    data.PrintHistogram(True, 20, "training data input", 0)
    if i == 4: break

# check a few slices of the trainingBatch output of the 1st training case (the density map)
data.OutData = bigOutputTrainingArray[0]
data.PrintSlice(2, MidSlice-1, False, 0)
data.PrintSlice(2, MidSlice,   False, 0)
data.PrintSlice(2, MidSlice+1, False, 0)

# check the trainingBatch output histogrram for a few training cases
for i in range(TrainingBatchSize):
    data.OutData = bigOutputTrainingArray[i]
    data.PrintHistogram(False, 20, "training data output", 0)
    if i == 4: break

# create a new file for recording the training and validation loss while training
history_file = outTxt
outfile = open(history_file, 'w')
outfile.write("input model: model %d, training %d\n" % (InModelNum, InTrainingNum))
outfile.write("output model: model %d, training %d\n" % (OutModelNum, OutTrainingNum))
outfile.write("TotalNumEpochs: %d\n" % (TotalNumEpochs))
outfile.write("NumEpochsPerWriteToFile: %d\n" % (NumEpochsPerWriteToFile))
outfile.write("NumEpochsPerMakeNewData: %d\n" % (NumEpochsPerMakeNewData))
outfile.write("NumEpochsPerViewOutput: %d\n" % (NumEpochsPerViewOutput))
outfile.write("TrainingBatchSize: %d\n" % (TrainingBatchSize))
outfile.write("ValidationBatchSize: %d\n" % (ValidationBatchSize))
outfile.write("BOX = %f : %f\n" % (BOX_MIN, BOX_MAX))
outfile.write("PIXEL_LEN = %f\n" % (PIXEL_LEN))
outfile.write("MinNumAtomsPerMolecule = %d\n" % (MinNumAtomsPerMolecule))
outfile.write("MaxNumAtomsPerMolecule = %d\n" % (MaxNumAtomsPerMolecule))
outfile.write("NumResiduesPerMolecule = %d\n" % (NumResiduesPerMolecule))
outfile.write("Learning Rate (LR) = %f\n" % (LR))
outfile.close()

# if it exists, load the NN model from disk. can be saved with model.save().
if InTrainingNum >= 1:
    model = load_model(inModel, custom_objects={'customLoss': customLoss})
    model.compile(optimizer=Adam(lr=LR), loss=customLoss)

#-------------------------------------------------------------------------------
# run the back propagation on the training data.
#-------------------------------------------------------------------------------
NumEpochs = 0
while NumEpochs < TotalNumEpochs:

    print("\niteration %d" % (NumEpochs))
    outfile = open(history_file, 'a')
    outfile.write("\niteration %d\n" % (NumEpochs))
    outfile.close()

    # run back prop on the training data
    hist = model.fit(bigInputTrainingArray, bigOutputTrainingArray, batch_size=1, epochs=NumEpochsPerWriteToFile, verbose=2)
    print(hist.history)

    outfile = open(history_file, 'a')
    for key,val in hist.history.items():
        outfile.write("%d: %s: [" % (NumEpochs, key))
        for index in range(len(val)-1):
            outfile.write("%e, " % (val[index]))
        outfile.write("%e" % (val[len(val)-1]))
        outfile.write("]\n")
    outfile.close()

    # save the model to disk. can be read with load_model().
    model.save(outModel)

    # calculate the loss for the validation data
    print("getting loss for validation data...")
    Loss = model.evaluate(bigInputValidationArray, bigOutputValidationArray, batch_size=1, verbose=2)
    print("Validation Loss = %e" %(Loss))

    outfile = open(history_file, 'a')
    outfile.write("validation loss = %e\n" % (Loss))
    outfile.close()

    NumEpochs += 1

    # periodically look at true and predicted output slices
    if (NumEpochs % NumEpochsPerViewOutput == 1):

        # get the NN output for the training data. only need to do this when we're visualizing it below.
        print("getting output for training data...")
        modelOutput = model.predict(bigInputTrainingArray, batch_size=1, verbose=2)

        # visualize the true output and training output for a few examples. (see "break" statement below).
        for i in range(TrainingBatchSize):

            # check what the true output looks like for the training data
            data.OutData = bigOutputTrainingArray[i]
            data.PrintSlice(2, MidSlice, False, 0)
            data.PrintHistogram(False, 20, 'True output', 0)

            # check what the NN output looks like for the training data
            data.OutData = modelOutput[i]
            data.PrintSlice(2, MidSlice, False, 0)
            data.PrintHistogram(False, 20, 'NN output', 0)

            # just for a few examples
            if i == 1: break

    # make new data every NumEpochsPerMakeNewData epochs
    if (NumEpochs % NumEpochsPerMakeNewData == 0):
        trainingBatch.makeABatch(allLinesTrain, AtomRadiusSqr, doRotation=True, batchType='training',
                                 OnlyCAlphas=UseOnlyCAlphas, NthCAlpha=EveryNthCAlpha,
                                 MinNumAtoms=MinNumAtomsPerMolecule, MaxNumAtoms=MaxNumAtomsPerMolecule,
                                 NumResidues=NumResiduesPerMolecule, AddSymmetryAtoms=AddSymmetryAtoms)
        (bigInputTrainingArray, bigOutputTrainingArray) = trainingBatch.makeBigArrays()

    # change the learning rate
    # model.compile(optimizer=Adam(lr=LR[i+1]), loss='mean_squared_error')
    # K.set_value(model.optimizer.lr, LR[i+1])
    # learning_rate = K.get_value(model.optimizer.lr)
    # print("Learning Rate = %e" % (learning_rate))

test = 1
