# author: David Hurwitz
# started: 1/25/19
#

from NN_raw_data import AllLines
from NN_readRawDataFiles import ReadFilesInList
from NN_connections import ProteinConnections
from NN_formatted_data import OneNNData
from NN_prepare_batch import OneBatch
from NN_misc import CalcAvgDistUnMatched, PIXEL_LEN
from keras.models import Model, load_model
from keras.layers import Conv3D, BatchNormalization, Input
from keras import optimizers
from keras import backend as K
from keras.optimizers import SGD, Adam

NumIterations = 175
NumEpochs = 10
TrainingBatchSize = 30
ValidationBatchSize = 10

ModelNum = 1
TrainingNum = 18
inModel =  "C:/Users/david/Documents/newff/results/NN/simulations/mol_05/NN_models/model_%02d/model_%02d_training_%02d.h5"  % (ModelNum, ModelNum, TrainingNum-1)
outModel = "C:/Users/david/Documents/newff/results/NN/simulations/mol_05/NN_models/model_%02d/model_%02d_training_%02d.h5"  % (ModelNum, ModelNum, TrainingNum)
outTxt =   "C:/Users/david/Documents/newff/results/NN/simulations/mol_05/NN_models/model_%02d/model_%02d_training_%02d.txt" % (ModelNum, ModelNum, TrainingNum)

# the input and output radius will be 2.0 for NumBigRadius iterations, then 1.5 for the remainder
NumBigRadius = 0
InRadii  = [2.0*PIXEL_LEN]*NumBigRadius + [1.00*PIXEL_LEN]*(NumIterations-NumBigRadius)
OutRadii = [2.0*PIXEL_LEN]*NumBigRadius + [1.00*PIXEL_LEN]*(NumIterations-NumBigRadius)

# set the learning rate of each iteration
LR = [3e-5]*NumIterations
#LR = []
#Rate = 1e-4
#Decay = 0.98
#for i in range(NumIterations):
#    LR.append(Rate)
#    Rate = Rate * Decay

#-----------------------------------------------------------
# read the CSV files that have the MD simulation data
# allLinesTrain has the raw training data
# allLinesValidate has the raw validation data
#-----------------------------------------------------------
CSV_Path = "C:/Users/david/Documents/newff/results/NN/simulations/mol_05/csv_files/"
FileListTrain = "sim_files_000_to_029.txt"       # 000_to_004 or 000_to_029
FileListValidate = "sim_files_030_to_039.txt"    # 005_to_009 or 030_to_039
print("reading the training files listed in: " + CSV_Path + FileListTrain)
allLinesTrain = AllLines()
ReadFilesInList(allLinesTrain, CSV_Path, FileListTrain)
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
# make a OneNNData to get array sizes for the NN
#-------------------------------------------------------------------------------
data = OneNNData(SizedForNN=True)  # default InData size is larger for extra workspace
inShape = data.InData.shape
outShape = data.OutData.shape

#-------------------------------------------------------------------------------
# make the Keras Functional API model.
#-------------------------------------------------------------------------------
# input shape is: (?, 100, 100, 100, 16)
input =  Input(shape=data.InData.shape)
L01 =    Conv3D(18, 5, activation='relu', padding='same', kernel_initializer='he_normal') (input)
L02 =    Conv3D(18, 5, activation='relu', padding='same', kernel_initializer='he_normal') (L01)
L03 =    Conv3D(18, 5, activation='relu', padding='same', kernel_initializer='he_normal') (L02)
L04 =    Conv3D(18, 5, activation='relu', padding='same', kernel_initializer='he_normal') (L03)
L05 =    Conv3D(18, 5, activation='relu', padding='same', kernel_initializer='he_normal') (L04)
L06 =    Conv3D(18, 5, activation='relu', padding='same', kernel_initializer='he_normal') (L05)
L07 =    Conv3D(18, 5, activation='relu', padding='same', kernel_initializer='he_normal') (L06)
L08 =    Conv3D(18, 5, activation='relu', padding='same', kernel_initializer='he_normal') (L07)
output = Conv3D( 1, 5, activation='tanh', padding='same', kernel_initializer='he_normal') (L08)  # padding = 'same'/'valid', activation = 'sigmoid'/'tanh'
model =  Model(inputs=[input], outputs=[output])
print(model.summary(line_length=150))
model.compile(optimizer=Adam(lr=LR[0]), loss='mean_squared_error')   # optimizer = SGD(lr=1e-2) / Adam(lr=4e-4)

CurInRadius = InRadii[0]
CurOutRadius = OutRadii[0]
InRadiusSqr = CurInRadius*CurInRadius
OutRadiusSqr = CurOutRadius*CurOutRadius

#-------------------------------------------------------------------------------
# make a batch of formatted data for training
# make a batch of formatted data for validation
#-------------------------------------------------------------------------------
trainingBatch = OneBatch(TrainingBatchSize)
trainingBatch.makeABatch(allLinesTrain, Connections, InRadiusSqr, OutRadiusSqr, doRotation=True, batchType='training')
(bigInputTrainingArray, bigOutputTrainingArray) = trainingBatch.makeBigArrays(borderDecrease=0, combineDensityMaps=True)

validationBatch = OneBatch(ValidationBatchSize)
validationBatch.makeABatch(allLinesValidate, Connections, InRadiusSqr, OutRadiusSqr, doRotation=True, batchType='validation')
(bigInputValidationArray, bigOutputValidationArray) = validationBatch.makeBigArrays(borderDecrease=0, combineDensityMaps=True)

# see if the trainingBatch output looks right for the mid-slice of the 1st training case
# print the training-data output histogram
data.SetOutData(bigOutputTrainingArray[0])
data.PrintSlice(2, 50, 'A', 0)
data.PrintHistogram(0, 'A', 20, 'training-data output')

# see if the validationBatch output looks right
# print the validation-data output histogram
data.SetOutData(bigOutputValidationArray[0])
data.PrintSlice(2, 50, 'A', 0)
data.PrintHistogram(0, 'A', 20, 'validation-data output')

# create a new file for recording the training and validation loss while training
history_file = outTxt
outfile = open(history_file, 'w')
outfile.close()

# if it exists, load the NN model from disk. can be saved with model.save().
if TrainingNum > 1:
    model = load_model(inModel)

#-------------------------------------------------------------------------------
# run the back propagation on the training data.
#-------------------------------------------------------------------------------
for i in range(NumIterations):

    print("iteration = %d, in-radius = %f, out-radius = %f, learning-rate = %e" %(i, InRadii[i], OutRadii[i], LR[i]))

    # run back prop on the training data
    hist = model.fit(bigInputTrainingArray, bigOutputTrainingArray, batch_size=1, epochs=NumEpochs)
    print(hist.history)

    # save the model to disk. can be read with load_model().
    model.save(outModel)

    # calculate the loss for the validation data
    print("getting loss for validation data...")
    Loss = model.evaluate(bigInputValidationArray, bigOutputValidationArray, batch_size=1)
    print("Validation Loss = %e" %(Loss))

    # save the training-loss and validation-loss history to file
    # hist.history is a dictionary. key is a string ('loss'), val is a list of floats (the losses).
    outfile = open(history_file, 'a+')
    for key,val in hist.history.items():
        outfile.write("%d: %s: [" % (i, key))
        for index in range(len(val)-1):
            outfile.write("%e, " % (val[index]))
        outfile.write("%e" % (val[len(val)-1]))
        outfile.write("]\n")
    outfile.write("validation loss = %e\n" % (Loss))
    outfile.close()

    # get the NN output for the training data
    print("getting output for training data...")
    modelOutput = model.predict(bigInputTrainingArray, batch_size=1, verbose=1)

    # check what the NN output looks like
    data.SetOutData(modelOutput[0])
    #data.PrintSlice(2, 50, 'A', 0)
    data.PrintHistogram(0, 'A', 20, 'NN output')

    # no need to do prep work for the next iteration on the last iteration
    if i == (NumIterations-1):
        break

    # make new data for next iteration
    InRadiusSqr = InRadii[i+1] * InRadii[i+1]
    OutRadiusSqr = OutRadii[i+1] * OutRadii[i+1]
    trainingBatch.makeABatch(allLinesTrain, Connections, InRadiusSqr, OutRadiusSqr, doRotation=True, batchType='training')
    (bigInputTrainingArray, bigOutputTrainingArray) = trainingBatch.makeBigArrays(borderDecrease=0, combineDensityMaps=True)

    #-----------------------------------------------------------------------------------------------#
    #                the section below is currently not needed                                      #
    #                it's only used when the atom radius is changing during training                #
    #-----------------------------------------------------------------------------------------------#
    # use -1.0 to signal that InRadius or OutRadius did not change
    NewInRadius = InRadii[i+1]
    NewOutRadius = OutRadii[i+1]
    InRadiusSqr = NewInRadius * NewInRadius
    OutRadiusSqr = NewOutRadius * NewOutRadius
    if CurInRadius == NewInRadius:
        InRadiusSqr = -1.0
    if CurOutRadius == NewOutRadius:
        OutRadiusSqr = -1.0

    # only need to remake the density maps if either the InRadius or OutRadius changed
    if (InRadiusSqr > 0.0) or (OutRadiusSqr > 0.0):

        # remake the output density maps (InRadiusSqr = -1.0 means the input maps aren't remade)
        print("remaking training batch...")
        trainingBatch.ReMakeBatch(InRadiusSqr, OutRadiusSqr, batchType='training')
        print("remaking validation batch...")
        validationBatch.ReMakeBatch(InRadiusSqr, OutRadiusSqr, batchType='validation')

        # get the combined density output maps
        (bigInputTrainingArray, bigOutputTrainingArray) = trainingBatch.makeBigArrays(borderDecrease=0, combineDensityMaps=True)
        (bigInputValidationArray, bigOutputValidationArray) = validationBatch.makeBigArrays(borderDecrease=0, combineDensityMaps=True)

        # check what the training outputs look like
        data.SetOutData(bigOutputTrainingArray[0])
        #data.PrintSlice(2, 50, 'A', 0)
        data.PrintHistogram(0, 'A', 20, 'training-data output')

        # check what the validation outputs look like
        data.SetOutData(bigOutputValidationArray[0])
        #data.PrintSlice(2, 50, 'A', 0)
        data.PrintHistogram(0, 'A', 20, 'validation-data output')

    CurInRadius = NewInRadius
    CurOutRadius = NewOutRadius
    #-----------------------------------------------------------------------------------------------#
    #                the section above is currently not needed                                      #
    #                it's only used when the atom radius is changing during training                #
    #-----------------------------------------------------------------------------------------------#

    # change the learning rate
    # model.compile(optimizer=Adam(lr=LR[i+1]), loss='mean_squared_error')
    K.set_value(model.optimizer.lr, LR[i+1])
    learning_rate = K.get_value(model.optimizer.lr)
    print("Learning Rate = %e" % (learning_rate))

    # compare actual output positions to those estimated from the NN output
    # data = trainingBatch.getOneItem(0)                                    # put the 1st training data in a OneNNData
    # (ktypes, kpos) = data.GetRawData(0)                                   # get the exact output (time-step 0)
    # (etypes, epos) = data.EstimateAtomPositionsFromDensityMaps(0)         # estimate atom positions from density maps (time-step 0)
    # (avg, max) = CalcAvgDistUnMatched(kpos, epos)                         # calculate avg dist between exact and estimate

test = 1
