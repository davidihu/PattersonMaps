# author: David Hurwitz
# started: 3/15/18
#

import csv
import numpy as np
import pandas as pd
from NN_raw_data import ALine
from NN_raw_data import AllLines
from NN_raw_data import STEP_NUM_INC
from NN_misc import AVERAGE_POSITION

#-------------------------------------------------------------------------------
# Read all files specifed in Path+FileList into allLines
# once all the data is read, pre-calculate all the protein centers
#-------------------------------------------------------------------------------
def ReadFilesInList(allLines, Path, FileList):
    # read all the simulation data
    with open((Path+FileList), 'r') as list:
        for File in list:
            File = File.rstrip('\n')
            print("reading file: " + Path + File)
            ReadAFile(allLines, Path, File)
    # calculate the protein-center for each time-step of each simulation of each molecule
    allLines.CalcAndSaveAllProteinCenters(AVERAGE_POSITION)

#-------------------------------------------------------------------------------
# read a single simulation file into allLines
# assumes the fileName is molxxx_simyyy...
#-------------------------------------------------------------------------------
def ReadAFile(allLines, Path, File):
    molNum = int(File[3:6])
    simNum = int(File[10:13])
    with open((Path + File), 'r') as theFile:
        reader = csv.reader(theFile)
        next(reader)  # ignore the header lines.
        for row in reader:
            Line = ALine(molNum, simNum, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7])
            allLines.addLine(Line)

#-------------------------------------------------------------------------------
# read a single simulation file into allLines
# assumes the fileName is molxxx_simyyy...
# I'm trying to speedup ReadAFile by using pandas.read_csv
# But this actually turns out to be much slower than ReadAFile()
# DON'T USE. This is ~20x slower than ReadAFile.
#-------------------------------------------------------------------------------
def ReadAFile2(allLines, Path, File):
    molNum = int(File[3:6])
    simNum = int(File[10:13])
    df = pd.read_csv(Path+File)
    for index, row in df.iterrows():
        Line = ALine(molNum, simNum, row[0], row[1], row[2], row[5], row[6], row[7])
        allLines.addLine(Line)

