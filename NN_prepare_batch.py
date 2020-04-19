# author: David Hurwitz
# started: 3/28/18
#

import numpy as np
import random
from NN_raw_data import ALine
from NN_raw_data import AllLines
from NN_formatted_data import OneNNData, NUM_IN_CHANNELS, NUM_OUT_CHANNELS, NUM_PIXELS_1D
from NN_misc import STEP_NUM_INC, SMALL_NUM, VERY_SMALL_NUM
from numpy import float32

# for NN batch data
class OneBatch:

    #-------------------------------------------------------------------------------------
    # initialize all OneNNData objects needed for presenting the batch data to the NN.
    # this needs to be executed just one time.
    #-------------------------------------------------------------------------------------
    def __init__(self, numInBatch):
        self.trainingExamples = []
        for i in range(numInBatch):
            aTrainingExample = OneNNData()
            self.trainingExamples.append(aTrainingExample)

    #-------------------------------------------------------------------------------------
    # the number of training examples in the batch
    #-------------------------------------------------------------------------------------
    def getNumInBatch(self):
        return(len(self.trainingExamples))

    #-------------------------------------------------------------------------------------
    # for each set of atom positions in this batch (one for each time step),
    # remake the density maps.
    #-------------------------------------------------------------------------------------
    def ReMakeBatch(self, InData, AtomRadiusSqr):
        # for each object in the batch
        for i in range(len(self.trainingExamples)):
            self.trainingExamples[i].ReMakeMap(InData, AtomRadiusSqr)

    #-------------------------------------------------------------------------------------
    # for each array in this batch, make a new training example.
    # this can be called multiple times because each call to SetOneTrainingExample clears the NN data
    #-------------------------------------------------------------------------------------
    def makeABatch(self, allLines, AtomRadiusSqr, doRotation, batchType,
                   printNumPixelsTouchedInOutputDensity=False, DisplacementNoise=0.0,
                   OnlyCAlphas=False, NthCAlpha=1,
                   MinNumAtoms=-1, MaxNumAtoms=-1, NumResidues=-1, AddSymmetryAtoms=False):

        # for each object in the batch
        for i in range(len(self.trainingExamples)):

            Found = False
            while not Found:

                # pick a random line from the raw-data file
                numLines = allLines.getNumLines()
                lineIndex = random.randrange(0, numLines-1)
                aLine = allLines.getALine(lineIndex)

                # make the NN-formatted arrays
                molNum = aLine.molNum
                simNum = aLine.simNum
                stepNum = aLine.stepNum

                # for some reason, the last step in each simulation crashes.
                # for now anyway, i'm not fixing the problem, just avoiding it.
                if stepNum != 100000:
                    Found = True

            if i % 100 == 0:
                print("%3d: making %s data for molNum: %d, simNum: %d, stepNum: %d" %(i+1, batchType, molNum, simNum, stepNum))
            if i % 100 == 1:
                print("...")

            # this training example will have between MinNumAtoms and MaxNumAtoms atoms
            NumAtoms = -1
            if MinNumAtoms > 0:
                NumAtoms = random.randint(MinNumAtoms, MaxNumAtoms)

            self.trainingExamples[i].SetOneTrainingData(allLines, molNum, simNum, stepNum, AtomRadiusSqr, doRotation,
                                                        printNumPixelsTouchedInOutputDensity, DisplacementNoise,
                                                        OnlyCAlphas, NthCAlpha, NumAtoms, NumResidues, AddSymmetryAtoms,
                                                        batchType)

    #-------------------------------------------------------------------------------------
    # make the big input and output arrays
    # BorderDecrease is the amount to reduce the output arrays all around.
    # e.g.: bigOutputArray = 10 x 100 x 100 x 100 x 4
    #       borderDecrease = 1
    #       bigOutputArray becomes 10 x 98 x 98 x 98 x 4
    #
    # expansionSize > 0 is NOT USED.
    # Which means expandOne is NOT CALLED.
    #-------------------------------------------------------------------------------------
    def makeBigArrays(self):
        bigInputArray = self.makeBigInputArray()
        bigOutputArray = self.makeBigOutputArray()
        return(bigInputArray, bigOutputArray)

    #-------------------------------------------------------------------------------------
    # make one big concatenated input array from the inputData arrays in each OneNNData
    #
    # Note that though OneNNData has NUM_IN_CHANNELS in it's InData array,
    # only NUM_IN_DATA_CHANNELS are used here when making the bigInputArray that is
    # returned by this routine. The other channels are used for workspace.
    #-------------------------------------------------------------------------------------
    def makeBigInputArray(self):
        # collect the input arrays
        arrays = []
        for i in range(len(self.trainingExamples)):
            # debugging:
            # shape1 = self.trainingExamples[i].InData.shape
            # shape2 = self.trainingExamples[i].InData[:,:,:,0:NUM_IN_DATA_CHANNELS].shape
            # old way of doing it:
            # arrays.append(self.trainingExamples[i].InData)
            # new way of doing it: don't present the 4 channels used as workspace to the NN:
            arrays.append(self.trainingExamples[i].InData[:,:,:,0:NUM_IN_CHANNELS])
        # make a big stacked input array from them
        bigInputArray = np.stack(arrays, axis=0)
        #print("bigInputArray.shape = ", bigInputArray.shape)
        return(bigInputArray)

    #-------------------------------------------------------------------------------------
    # make one big concatenated output array from the outputData arrays in each OneNNData
    #-------------------------------------------------------------------------------------
    def makeBigOutputArray(self):
        # collect the output arrays
        arrays = []
        for i in range(len(self.trainingExamples)):
            arrays.append(self.trainingExamples[i].OutData)
        bigOutputArray = np.stack(arrays, axis=0)
        #print("bigOutputArray.shape = ", bigOutputArray.shape)
        return(bigOutputArray)

    #-------------------------------------------------------------------------------------
    # gives one item in the batch
    #-------------------------------------------------------------------------------------
    def getOneItem(self, index):
        assert(index < self.getNumInBatch())
        return(self.trainingExamples[index])

    #-------------------------------------------------------------------------------------
    # sets one item in the batch
    #-------------------------------------------------------------------------------------
    def setOneItem(self, index, data):
        assert(index < self.getNumInBatch())
        self.trainingExamples[index] = data
