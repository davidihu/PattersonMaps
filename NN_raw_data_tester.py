# author: David Hurwitz
# started: 3/14/18
#
# tests of NN_raw_data.py

import numpy as np
import time
from NN_readRawDataFiles import ReadFilesInList
from NN_raw_data import AllLines
from NN_misc import CalcCenter, CalcCenter1, CalcCenter2
from NN_misc import Get3RandomAngles, MakePDB, MakeSimilarPDB, RotateProtein, PI, CalcRMSD
from NN_misc import TrackMinsAndMaxs
from NN_ARotationMatrix import ARotationMatrix
from numpy import float32

molNum = 5
CSV_Path = "C:/Users/david/Documents/newff/results/NN/simulations/mol_05/csv_files/"
FileList  = "sim_files_000_to_004.txt"
FileList2 = "sim_files_005_to_009.txt"

# read in all the CSV files with the MD simulation data for simulations 0 - 4.
# the data is stored in an AllLines object
startTime = time.time()
Lines = AllLines()
ReadFilesInList(Lines, CSV_Path, FileList)
elapsedTime = time.time() - startTime

# read in all the CSV files with the MD simulation data for simulations 5 - 9.
# the data is stored in an AllLines object
#startTime = time.time()
#Lines2 = AllLines()
#ReadFilesInList(Lines2, CSV_Path, FileList2)
#elapsedTime2 = time.time() - startTime

# check a few lines of raw data
OneLine = Lines.GetALine(molNum, 0, 0, 0)
assert(OneLine.atomType == "N")
OneLine = Lines.GetALine(molNum, 0, 0, 1)
assert(OneLine.atomType == "C")
OneLine = Lines.GetALine(molNum, 1, 2010, 10)
assert(OneLine.atomType == "C")
OneLine = Lines.GetALine(molNum, 1, 2010, 11)
assert(OneLine.atomType == "C")
assert(OneLine.molNum == molNum)
assert(OneLine.simNum == 1)
assert(OneLine.stepNum == 2010)
assert(OneLine.atomNum == 11)
OneLine = Lines.GetALine(molNum, 1, 2010, 12)
assert(OneLine.atomType == "O")

# check that numResidues is calculated correctly
NumAtoms1 = Lines.GetNumAtoms(molNum, 0, 0)
NumAtoms2 = Lines.GetNumAtoms(molNum, 0, 2000)
assert(NumAtoms2 == NumAtoms1)
NumAtoms3 = Lines.GetNumAtoms(molNum, 0, 2000)
assert(NumAtoms3 == NumAtoms1)
NumAtoms4 = Lines.GetNumAtoms(molNum, 0, 2100)
assert(NumAtoms4 == 0)
NumAtoms5 = Lines.GetNumAtoms(molNum, 4, 2010)
assert(NumAtoms5 == NumAtoms1)

# check that numTimeSteps is calculated correctly
NumSteps1 = Lines.GetNumTimeSteps(molNum, 0)
NumSteps2 = Lines.GetNumTimeSteps(molNum, 3)
assert(NumSteps2 == NumSteps1)
NumSteps3 = Lines.GetNumTimeSteps(molNum, 4)
assert(NumSteps3 == NumSteps1)
NumSteps4 = Lines.GetNumTimeSteps(molNum, 5)
#assert(NumSteps4 == 0)  # depends on the number of simulations read in

# check that numSimulations is calculated correctly
NumSimulations = Lines.GetNumSimulations(molNum)
assert(NumSimulations > 0 and NumSimulations <= 40)
NumSimulations = Lines.GetNumSimulations(molNum+1)
assert(NumSimulations == 0)

# check that numMolecules is calculated correctly
NumMolecules = Lines.GetNumMolecules()
assert(NumMolecules == 1)

# check that pre-calculated protein centers were calculated properly
# these are protein centers of raw-data. but they shouldn't move much over a few time steps.
proteinCenter1 = Lines.GetProteinCenter(molNum, 0, 0)
proteinCenter2 = Lines.GetProteinCenter(molNum, 0, 49)
dist = np.linalg.norm(proteinCenter1 - proteinCenter2)
assert(dist < 1.0)

# get data for one time-step. calculate its center. it should be the same as the pre-calculated center.
atomPositions = Lines.GetProteinCoordinates(molNum, 0, 49)    # same time-step as proteinCenter2 above
Center = CalcCenter(atomPositions)
dist = np.linalg.norm(proteinCenter2 - Center)
assert(dist < 0.01)

# get translated data for one time-step.
# in this case, it should be translated s.t. its new center is at (0,0,0)
atomPositions = Lines.GetProteinCoordinatesTranslated(molNum, 0, 49, Center)
NewCenter = CalcCenter(atomPositions)
Zero = np.zeros(shape=[3], dtype=float32)
dist = np.linalg.norm(NewCenter - Zero)
assert(dist < 0.01)

curMinsAndMaxs = (999, -999, 999, -999, 999, -999)
for i in range(int(1E4)):   # can be 1E5, 1E6 or 1E7 for more accurate measurments
    newMinsAndMaxs = Lines.GetBoundingBoxOfRandomProteinCenteredAndRotated()
    curMinsAndMaxs = TrackMinsAndMaxs(i, curMinsAndMaxs, newMinsAndMaxs)

# test of CalcRMSD
# Vec1 -> Vec2 = (0.0, 0.1, 0.0)
# RMSD should be sqrt(0**2 + 0.1**2 + 0**2) = 0.1 / sqrt(3)
Vec1 = np.array([0.0, 1.0, 0.0], dtype=float32)
Vec2 = np.array([0.0, 1.1, 0.0], dtype=float32)
RMSD = CalcRMSD(Vec1, Vec2)
assert(abs(RMSD - 0.1 / np.sqrt(3)) < 0.001)

# test CalcRMSD on a few proteins
atomPositions1 = Lines.GetProteinCoordinatesTranslated(molNum, 0, 2000, Center)
atomPositions2 = Lines.GetProteinCoordinatesTranslated(molNum, 0, 2010, Center)
atomPositions3 = Lines.GetProteinCoordinatesTranslated(molNum, 0, 2020, Center)
atomPositions4 = Lines.GetProteinCoordinatesTranslated(molNum, 0, 2030, Center)
print(atomPositions1.shape)
print(atomPositions2.shape)
print(atomPositions3.shape)
print(atomPositions4.shape)
RMSD12 = CalcRMSD(atomPositions1, atomPositions2)
RMSD13 = CalcRMSD(atomPositions1, atomPositions3)
RMSD14 = CalcRMSD(atomPositions1, atomPositions4)
RMSD23 = CalcRMSD(atomPositions2, atomPositions3)
RMSD24 = CalcRMSD(atomPositions2, atomPositions4)
RMSD34 = CalcRMSD(atomPositions3, atomPositions4)
assert(RMSD12 < 1 and RMSD13 < 1 and RMSD12 < RMSD13)
assert(RMSD14 < 1 and RMSD13 < RMSD14)
assert(RMSD23 < 1 and RMSD24 < 1 and RMSD23 < RMSD24)
assert(RMSD34 < 1)

# test rotation matrix. rotate protein (PI/3/1.0001) 6 times = 359.64 degrees
# the rmsd between before-rotation and after-rotation should be non-zero but pretty small
atomPositions = Lines.GetProteinCoordinatesTranslated(molNum, 0, 2000, Center)
Axis = np.array([1.0, 1.0, 1.0], dtype=float32)
RotationMatrix = ARotationMatrix(Axis, PI/3/1.0001)
rotatedAtomPositions = list(atomPositions)
for j in range(6):
    for i in range(len(atomPositions)):
        rotatedAtomPositions[i] = np.matmul(RotationMatrix.m_matrix, rotatedAtomPositions[i])
RMSD = CalcRMSD(atomPositions, rotatedAtomPositions)
assert(RMSD < 0.02)

# test random-rotation of a protein, and MakeSimilarPDB.
# do visual inspection of the pdb files. check that the 2nd seems a random orientation of the 1st
template = "C:/Users/david/Documents/newff/results/NN/simulations/mol_05/pdb_files/tmp200600.pdb"
Center = Lines.GetProteinCenter(molNum, 0, 2000)
atomPositions = Lines.GetProteinCoordinatesTranslated(molNum, 0, 2000, Center)
MakeSimilarPDB(template, "rotationTest0.pdb", atomPositions)
(Angle1, Angle2, Angle3) = Get3RandomAngles()
rotatedAtomPositions = RotateProtein(atomPositions, Angle1, Angle2, Angle3)
MakeSimilarPDB(template, "rotationTest1.pdb", rotatedAtomPositions)

stop = True
