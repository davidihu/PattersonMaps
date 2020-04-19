# author: David Hurwitz
# started: 3/14/18
#

import numpy as np
import random
from NN_misc import makeKey
from NN_misc import STEP_NUM_INC
from NN_misc import Get3RandomAngles, RotateProtein
from numpy import float32

CAlphaIndices = [1, 9, 17, 29, 37, 46, 60, 68, 77, 85, 89, 93, 100, 106, 112, 116, 127, 134, 141, 148]

#-------------------------------------------------------------------------------
# ALine stores one line of data from one simulation.
#-------------------------------------------------------------------------------
class ALine:
    # CA = CAlpha; SC = SideChain
    def __init__(self, molNum, simNum, stepNum, atomNum, atomType, resNum, resType, atom_x, atom_y, atom_z):
        self.molNum = int(molNum)
        self.simNum = int(simNum)
        self.stepNum = int(stepNum)
        self.atomNum = int(atomNum)
        self.atomType = str(atomType)
        self.resNum = int(resNum)
        self.resType = str(resType)
        self.atomPos = np.array([float32(atom_x), float32(atom_y), float32(atom_z)], dtype=float32)

#-------------------------------------------------------------------------------
# AllLines stores all lines from all simulations.
# allLinesLookup{} provides quick access to time-step data in AllLines
# The key for allLinesLookup is a unique string made from molNum, simNum, and stepNum.
# The value for allLinesLookup is the index in AllLines to the 1st residue of a simulation.
# Lines must be added to AllLines in order of: 1) atom-num, 2) step-num, 3) sim-num, 4) mol-num
# This class also stores pre-calculated protein centers based on raw data.
#-------------------------------------------------------------------------------
class AllLines:
    def __init__(self):
        self.allLines = []                # allLines from all simulations
        self.allLinesLookup = {}          # dict: key = id from (molNum, simNum, stepNum), val = index of res in allLines
        self.prev_stepNum = -1            # when reading simulation files, for tracking when simulation number changes
        self.proteinCenters = []          # list of protein centers
        self.proteinCentersLookup = {}    # dict: key = id from (molNum, simNum, stepNum), val = index into proteinCenters

    #-------------------------------------------------------------------------------------------------------
    # save allLinesLookup info
    #-------------------------------------------------------------------------------------------------------
    def saveLookupInfo(self, Line):
        key = makeKey(Line.molNum, Line.simNum, Line.stepNum)
        self.allLinesLookup[key] = len(self.allLines)

    #-------------------------------------------------------------------------------------------------------
    # adds ALine (info for one residue) to AllLines (info for all simulations)
    # a list won't work as a key to a dictionary, so, I make a unique string from 3 ints to use as a key
    #-------------------------------------------------------------------------------------------------------
    def addLine(self, Line):
        # when we start a new time-step, save the allLines index for quick lookup
        if (Line.stepNum != self.prev_stepNum):
            self.prev_stepNum = Line.stepNum
            key = makeKey(Line.molNum, Line.simNum, Line.stepNum)
            self.allLinesLookup[key] = len(self.allLines)
        self.allLines.append(Line)

    #-------------------------------------------------------------------------------------------------------
    # get the total number of lines of raw data
    #-------------------------------------------------------------------------------------------------------
    def getNumLines(self):
        numLines = len(self.allLines)
        return(numLines)

    #-------------------------------------------------------------------------------------------------------
    # get one particular line from the raw data
    #-------------------------------------------------------------------------------------------------------
    def getALine(self, lineIndex):
        return(self.allLines[lineIndex])

    #-------------------------------------------------------------------------------------------------------
    # get starting index in AllLines for: molNum, simNum, stepNum
    #-------------------------------------------------------------------------------------------------------
    def GetStartIndex(self, molNum, simNum, stepNum):
        key = makeKey(molNum, simNum, stepNum)
        if key in self.allLinesLookup:
            index = self.allLinesLookup[key]
            return(index)
        else:
            return(None)

    #-------------------------------------------------------------------------------------------------------
    # return true if data is present for (molNum, simNum, stepNum)
    #-------------------------------------------------------------------------------------------------------
    def IsValid(self, molNum, simNum, stepNum):
        key = makeKey(molNum, simNum, stepNum)
        if key in self.allLinesLookup:
            return(True)
        else:
            return(False)

    #-------------------------------------------------------------------------------------------------------
    # return ALine from AllLines for: molNum, simNum, stepNum, resNum
    #-------------------------------------------------------------------------------------------------------
    def GetALine(self, molNum, simNum, stepNum, atomNum):
        key = makeKey(molNum, simNum, stepNum)
        if key in self.allLinesLookup:
            index = self.allLinesLookup[key]
            return(self.allLines[index + atomNum])
        else:
            return(None)

    #-------------------------------------------------------------------------------------------------------
    # get molNum of the first molecule
    #-------------------------------------------------------------------------------------------------------
    def GetMolNumStart(self):
        aLine = self.allLines[0]
        return(aLine.molNum)

    #-------------------------------------------------------------------------------------------------------
    # get molNum of the last molecule
    #-------------------------------------------------------------------------------------------------------
    def GetMolNumStop(self):
        index = len(self.allLines) - 1
        aLine = self.allLines[index]
        return(aLine.molNum)

    #-------------------------------------------------------------------------------------------------------
    # get the number of molecules in the simulations
    #-------------------------------------------------------------------------------------------------------
    def GetNumMolecules(self):
        numMolecules = self.GetMolNumStop() - self.GetMolNumStart() + 1
        return(numMolecules)

    #-------------------------------------------------------------------------------------------------------
    # get the number of the 1st simulation
    #-------------------------------------------------------------------------------------------------------
    def GetStartingSimNum(self):
        # get the simNum of the 1st line
        Line = self.getALine(0)
        startingSimNum = Line.simNum
        return(startingSimNum)

    #-------------------------------------------------------------------------------------------------------
    # get the number of simulations for a given molecule
    #-------------------------------------------------------------------------------------------------------
    def GetNumSimulations(self, molNum):
        # get the simNum of the 1st line
        startingSimNum = self.GetStartingSimNum()
        # check that there's a molNum simulation
        key = makeKey(molNum, startingSimNum, 0)
        if key not in self.allLinesLookup:
            return(0)
        # see if there's a key for the next molecule
        key = makeKey(molNum+1, 0, 0)
        if key in self.allLinesLookup:
            # look at the line in allLines just prior to the start of the next molecule for the number of simulations
            index = self.allLinesLookup[key] - 1
            aLine = self.allLines[index]
            return(aLine.simNum + 1 - startingSimNum)
        else:
            # guess that we're at the last molecule. return the number of simulations in the last line
            index = len(self.allLines) - 1
            aLine = self.allLines[index]
            return(aLine.simNum + 1 - startingSimNum)

    #-------------------------------------------------------------------------------------------------------
    # get the number of time-steps for a given (molecule, simulation)
    #-------------------------------------------------------------------------------------------------------
    def GetNumTimeSteps(self, molNum, simNum):
        # check that there's a (molNum, simNum) simulation
        key = makeKey(molNum, simNum, 0)
        if key not in self.allLinesLookup:
            return(0)
        # see if there's a key for the next simulation of this molecule
        key = makeKey(molNum, simNum+1, 0)
        if key in self.allLinesLookup:
            # look at the line in allLines just prior to the start of the next simulation for the number of time-steps
            index = self.allLinesLookup[key] - 1
            aLine = self.allLines[index]
            return(aLine.stepNum)
        else:
            # see if there's a key for the first simulation of the next molecule
            key = makeKey(molNum+1, 0, 0)
            if key in self.allLinesLookup:
                # look at the line in allLines just prior to the start of the next simulation for the number of time-steps
                index = self.allLinesLookup[key] - 1
                aLine = self.allLines[index]
                return(aLine.stepNum)
            else:
                # guess that we're at the very last simulation. return the number of time-steps in the last line
                index = len(self.allLines) - 1
                aLine = self.allLines[index]
                return(aLine.stepNum)

    #-------------------------------------------------------------------------------------------------------
    # count the number of atomType atoms in the protein
    #-------------------------------------------------------------------------------------------------------
    def GetNumAtomType(self, molNum, simNum, stepNum, atomType):
        Count = 0
        NumAtoms = self.GetNumAtoms(molNum, simNum, stepNum)
        for i in range(NumAtoms):
            Line = self.GetALine(molNum, simNum, stepNum, i)
            if (Line.atomType == atomType):
                Count = Count + 1
        return(Count)

    #-------------------------------------------------------------------------------------------------------
    # get the number of atoms for a given (molecule, simulation, time-step)
    #-------------------------------------------------------------------------------------------------------
    def GetNumAtoms(self, molNum, simNum, stepNum):
        # get the index into AllLines for (molNum, simNum, stepNum)
        key1 = makeKey(molNum, simNum, stepNum)
        index1 = index2 = 0
        if key1 in self.allLinesLookup:
            index1 = self.allLinesLookup[key1]
        else:
            return(0)
        # get the index into AllLines for (molNum, simNum, stepNum+STEP_NUM_INC)
        key2 = makeKey(molNum, simNum, stepNum + STEP_NUM_INC)
        if key2 in self.allLinesLookup:
            index2 = self.allLinesLookup[key2]
        else:
            # if the 2nd key isn't present, increment index1 until allLines[index2] has different (molNum, simNum, stepNum)
            index2 = index1 + 1
            while True:
                if (index2 == len(self.allLines)):
                    break
                aLine = self.allLines[index2]
                if (aLine.molNum != molNum or aLine.simNum != simNum or aLine.stepNum != stepNum):
                    break
                index2 = index2 + 1

            # if the 2nd key isn't present, try the first time-step of the next simulation
            #key2 = makeKey(molNum, simNum+1, 0)
            #if key2 in self.allLinesLookup:
            #    index2 = self.allLinesLookup[key2]
            #else:
            #    # if the 2nd key isn't present, try the first simulation of the first time-step of the next molecule
            #    key2 = makeKey(molNum+1, 0, 0)
            #    if key2 in self.allLinesLookup:
            #        index2 = self.allLinesLookup[key2]
            #    else:
            #        # if the 2nd key isn't present, guess that we're at the very last simulation
            #        index2 = len(self.allLines)
        # return the number of lines between allLinesLookup[key1] and allLinesLookup[key2]
        return(index2 - index1)

    #-------------------------------------------------------------------------------------------------------
    # calculate and return the average of the atom positions for one time-step of one simulation.
    #-------------------------------------------------------------------------------------------------------
    def CalcProteinCenter1(self, molNum, simNum, stepNum):
        NumAtoms = self.GetNumAtoms(molNum, simNum, stepNum)
        center = np.zeros(shape=[3], dtype=float32)

        # sum the protein's coordinates
        startIndex = self.GetStartIndex(molNum, simNum, stepNum)
        for i in range(NumAtoms):
            aLine = self.allLines[startIndex+i]
            center = center + aLine.atomPos

        center = center / float32(NumAtoms)
        return(center)

    #-------------------------------------------------------------------------------------------------------
    # rather than average the atom positions...
    # find the min & max on each axis, and take the center point.
    #-------------------------------------------------------------------------------------------------------
    def CalcProteinCenter2(self, molNum, simNum, stepNum):
        xMin = yMin = zMin =  999
        xMax = yMax = zMax = -999

        NumAtoms = self.GetNumAtoms(molNum, simNum, stepNum)
        center = np.zeros(shape=[3], dtype=float32)

        # check each atom position
        startIndex = self.GetStartIndex(molNum, simNum, stepNum)
        for i in range(NumAtoms):
            aLine = self.allLines[startIndex+i]
            if aLine.atomPos[0] > xMax:   xMax = aLine.atomPos[0]
            if aLine.atomPos[1] > yMax:   yMax = aLine.atomPos[1]
            if aLine.atomPos[2] > zMax:   zMax = aLine.atomPos[2]
            if aLine.atomPos[0] < xMin:   xMin = aLine.atomPos[0]
            if aLine.atomPos[1] < yMin:   yMin = aLine.atomPos[1]
            if aLine.atomPos[2] < zMin:   zMin = aLine.atomPos[2]

        center[0] = (xMin + xMax) / 2.0
        center[1] = (yMin + yMax) / 2.0
        center[2] = (zMin + zMax) / 2.0
        return(center)

    #-------------------------------------------------------------------------------------------------------
    # calculate the protein-center for each time-step of each simulation of each molecule
    #-------------------------------------------------------------------------------------------------------
    def CalcAndSaveAllProteinCenters(self, AveragePosition):
        molNumStart = self.GetMolNumStart()
        molNumStop = self.GetMolNumStop()
        for i in range(molNumStart, molNumStop+1):
            NumSimulations = self.GetNumSimulations(i)
            for j in range(NumSimulations):
                j = j + self.GetStartingSimNum()
                NumTimeSteps = self.GetNumTimeSteps(i, j)
                for k in range(0, NumTimeSteps):
                    key = makeKey(i, j, k)
                    if key in self.allLinesLookup:
                        self.CalcAndSaveProteinCenter(i, j, k, AveragePosition)

    #-------------------------------------------------------------------------------------------------------
    # calculate the protein center for a time-step.
    # add the protein center to the list of proteinCenters.
    # add a key to the proteinCentersLookup dictionary giving the position of the protein center in the list
    #-------------------------------------------------------------------------------------------------------
    def CalcAndSaveProteinCenter(self, molNum, simNum, stepNum, AveragePosition):
        if AveragePosition:
            center = self.CalcProteinCenter1(molNum, simNum, stepNum)
        else:
            center = self.CalcProteinCenter2(molNum, simNum, stepNum)
        key = makeKey(molNum, simNum, stepNum)
        self.proteinCentersLookup[key] = len(self.proteinCenters)
        self.proteinCenters.append(center)

    #-------------------------------------------------------------------------------------------------------
    # get the saved protein center for a time-step
    #-------------------------------------------------------------------------------------------------------
    def GetProteinCenter(self, molNum, simNum, stepNum):
        key = makeKey(molNum, simNum, stepNum)
        if key in self.proteinCentersLookup:
            index = self.proteinCentersLookup[key]
            return(self.proteinCenters[index])
        else:
            return(None)

    #-------------------------------------------------------------------------------------------------------
    # determine if AtomIndex is an Nth CAlpha
    # if Nth = 2, say, return True for the 0, 2, 4, ... elements.
    #-------------------------------------------------------------------------------------------------------
    def IsNthCAlpha(self, AtomIndex, Nth):
        if AtomIndex in CAlphaIndices:
            if CAlphaIndices.index(AtomIndex) % Nth == 0:
                return(True)
        return(False)

    #-------------------------------------------------------------------------------------------------------
    # get raw-data atom coordinates for one time-step
    #-------------------------------------------------------------------------------------------------------
    def GetProteinCoordinates(self, molNum, simNum, stepNum, OnlyCAlphas=False, NthCAlpha=1):
        key = makeKey(molNum, simNum, stepNum)
        if key in self.allLinesLookup:
            index = self.allLinesLookup[key]
            numAtoms = self.GetNumAtoms(molNum, simNum, stepNum)
            atomPositions = []
            for i in range(numAtoms):
                if (not OnlyCAlphas) or (OnlyCAlphas and self.IsNthCAlpha(i, NthCAlpha)):
                    Line = self.allLines[index + i]
                    atomPositions.append(Line.atomPos)
            return(atomPositions)
        else:
            return(None)

    #-------------------------------------------------------------------------------------------------------
    # get the atom types for one time-step.
    #-------------------------------------------------------------------------------------------------------
    def GetAtomTypes(self, molNum, simNum, stepNum, OnlyCAlphas=False, NthCAlpha=1):
        key = makeKey(molNum, simNum, stepNum)
        if key in self.allLinesLookup:
            index = self.allLinesLookup[key]
            numAtoms = self.GetNumAtoms(molNum, simNum, stepNum)
            atomTypes = []
            for i in range(numAtoms):
                if (not OnlyCAlphas) or (OnlyCAlphas and self.IsNthCAlpha(i, NthCAlpha)):
                    Line = self.allLines[index + i]
                    atomTypes.append(Line.atomType)
            return(atomTypes)
        else:
            return(None)

    #-------------------------------------------------------------------------------------------------------
    # get the residue types and residue numbers for one time-step.
    # each array that's returned has length num-atoms in the protein.
    #-------------------------------------------------------------------------------------------------------
    def GetResTypesAndResNumbers(self, molNum, simNum, stepNum, OnlyCAlphas=False, NthCAlpha=1):
        key = makeKey(molNum, simNum, stepNum)
        if key in self.allLinesLookup:
            index = self.allLinesLookup[key]
            numAtoms = self.GetNumAtoms(molNum, simNum, stepNum)
            resTypes = []
            resNumbers = []
            for i in range(numAtoms):
                if (not OnlyCAlphas) or (OnlyCAlphas and self.IsNthCAlpha(i, NthCAlpha)):
                    Line = self.allLines[index + i]
                    resTypes.append(Line.resType)
                    resNumbers.append(Line.resNum)
            return((resTypes, resNumbers))
        else:
            return(None)


    #-------------------------------------------------------------------------------------------------------
    # get translated CAlpha and side-chain-center coordinates for one time-step.
    # translate the raw coordinates by -Center (s.t. an atom at Center would -> (0,0,0))
    #-------------------------------------------------------------------------------------------------------
    def GetProteinCoordinatesTranslated(self, molNum, simNum, stepNum, Center, OnlyCAlphas, NthCAlpha):
        atomPositions = self.GetProteinCoordinates(molNum, simNum, stepNum, OnlyCAlphas, NthCAlpha)
        atomPositions = atomPositions - Center
        return(atomPositions)

    #-------------------------------------------------------------------------------------------------------
    # get bounding box for a protein conformation at a random time step.
    #-------------------------------------------------------------------------------------------------------
    def GetBoundingBoxOfRandomProteinCenteredAndRotated(self):

        # get a line with a valid Center
        GotOne = False
        while not GotOne:
            # pick a random line from the raw-data file
            lineIndex = random.randrange(0, self.getNumLines() - 1)
            # get the line
            aLine = self.getALine(lineIndex)
            # make sure there's a valid center
            Center = self.GetProteinCenter(aLine.molNum, aLine.simNum, aLine.stepNum)
            if Center is not None:
                GotOne = True

        # get the centered protein coordinates for that line
        atomPositions = self.GetProteinCoordinatesTranslated(aLine.molNum, aLine.simNum, aLine.stepNum, Center)

        # a check. mins and maxs should match in magnitude
        # xMin = atomPositions[:,0].min()
        # xMax = atomPositions[:,0].max()
        # yMin = atomPositions[:,1].min()
        # yMax = atomPositions[:,1].max()
        # zMin = atomPositions[:,2].min()
        # zMax = atomPositions[:,2].max()

        # rotate the protein coordinates
        (Angle1, Angle2, Angle3) = Get3RandomAngles()
        atomPositions = RotateProtein(atomPositions, Angle1, Angle2, Angle3)

        # get bounding box of protein
        xMin = atomPositions[:,0].min()
        xMax = atomPositions[:,0].max()
        yMin = atomPositions[:,1].min()
        yMax = atomPositions[:,1].max()
        zMin = atomPositions[:,2].min()
        zMax = atomPositions[:,2].max()

        return((xMin, xMax, yMin, yMax, zMin, zMax))
