# author: David Hurwitz
# started: 3/15/18
#

import numpy as np
import random, math
from NN_ARotationMatrix import ARotationMatrix
from NN_connections import ProteinConnections
from numpy import float32
from matplotlib import pyplot

# PIXEL_LEN = 0.32
# PIXEL_LEN = 0.36
# PIXEL_LEN = 0.40
# PIXEL_LEN = 0.80
PIXEL_LEN = 1.00
PIXEL_LEN_CUBED = PIXEL_LEN * PIXEL_LEN * PIXEL_LEN

#ATOM_RADIUS = 1.00 * PIXEL_LEN
#ATOM_RADIUS_SQR = ATOM_RADIUS * ATOM_RADIUS
#ATOM_RADIUS_CUBED = ATOM_RADIUS * ATOM_RADIUS * ATOM_RADIUS

# for speed, precalculate some stuff
PIXEL_LENS = [0.5, 0.5/2, 0.5/4, 0.5/8, 0.5/16, 0.5/32, 0.5/64, 0.5/128, 0.5/256, 0.5/512]
PIXEL_VOLUMES = []
PIXEL_VOLUMES_HALF = []
PIXEL_LENS_SQR = []
for i in range(len(PIXEL_LENS)):
    PIXEL_VOLUMES.append(PIXEL_LENS[i]*PIXEL_LENS[i]*PIXEL_LENS[i])
    PIXEL_VOLUMES_HALF.append(PIXEL_LENS[i]*PIXEL_LENS[i]*PIXEL_LENS[i]/2.0)
    PIXEL_LENS_SQR.append(PIXEL_LENS[i]*PIXEL_LENS[i])

# points on a grid. x = [0:1], y = [0:1], z = [0:1]
NUM_1D = 20
SPACING = (1.0/float(NUM_1D)) * PIXEL_LEN
GRID_PTS = np.zeros(shape=(NUM_1D*NUM_1D*NUM_1D, 3), dtype=float32)
# this can be slow since it's only done once
NUM_GRID_PTS = 0
for i in range(NUM_1D):
    x = i * SPACING + SPACING / 2
    for j in range(NUM_1D):
        y = j * SPACING + SPACING / 2
        for k in range(NUM_1D):
            z = k * SPACING + SPACING / 2
            GRID_PTS[NUM_GRID_PTS] = (x,y,z)
            NUM_GRID_PTS = NUM_GRID_PTS + 1

MINIMAL_GRID_PTS = np.zeros(shape=(8,3), dtype=float32)
MINIMAL_GRID_PTS[0] = (0,0,0)
MINIMAL_GRID_PTS[1] = (0,0,1)
MINIMAL_GRID_PTS[2] = (0,1,0)
MINIMAL_GRID_PTS[3] = (0,1,1)
MINIMAL_GRID_PTS[4] = (1,0,0)
MINIMAL_GRID_PTS[5] = (1,0,1)
MINIMAL_GRID_PTS[6] = (1,1,0)
MINIMAL_GRID_PTS[7] = (1,1,1)

PI = 3.141592653589
SMALL_NUM = 1E-6
VERY_SMALL_NUM = 1E-10

RES_TYPES = ["Ala", "Arg", "Asn", "Asp", "Cys",
             "Gln", "Glu", "Gly", "His", "Ile",
             "Leu", "Lys", "Met", "Phe", "Pro",
             "Ser", "Thr", "Trp", "Tyr", "Val"]
ATOM_TYPES = ["C","N","O","H","S"]

# number of time-steps between writes in the MD simulations
STEP_NUM_INC = 1

# true: use average residue position for center
# false: use center of minimum box bounding protein
AVERAGE_POSITION = False

PixelCompletelyOccupiedCount = 0
PixelCompletelyEmptyCount = 0
HalfVolumeCount = 0

#----------------------------------------------------------------------
# make a unique key from molNum, simNum, stepNum
#----------------------------------------------------------------------
def makeKey(molNum, simNum, stepNum):
    key = "molNum" + str(molNum) + "_simNum" + str(simNum) + "_stepNum" + str(stepNum)
    return (key)

#------------------------------------------------------------------------------
# reorder atomPositions2 so that the new order is a one-to-one correspondence
# with atomPositions1
# return the reordered atoms
#------------------------------------------------------------------------------
def MatchVectorOrder(atomPositions1, atomPositions2):

    sortedAtomPositions2 = []
    checkList = [0] * len(atomPositions2)

    # for each atom position in atomPositions1
    for i in range(len(atomPositions1)):
        smallest = 999.
        save_index = -1
        # look through atomPositions2 to find the nearest match
        for j in range(len(atomPositions2)):
            dist = np.linalg.norm(atomPositions1[i] - atomPositions2[j])
            if dist < smallest:
                smallest = dist
                save_index = j
        # save the nearest match in sortedAtomPositions
        sortedAtomPositions2.append(atomPositions2[save_index])
        # check that each atom position in atomPositions2 is only used once
        checkList[save_index] = checkList[save_index] + 1
        assert(checkList[save_index] <= 1)
    sortedAtomPositions2 = np.array(sortedAtomPositions2)
    return(sortedAtomPositions2)

#------------------------------------------------------------------------------
# similar to CalcAvgDist, but the correspondence between atoms is unknown.
# Therefore, it is first determined here, then CalcAvgDist is called.
# This routine naively assumes that each atom has one unique nearest neighbor.
#------------------------------------------------------------------------------
def CalcAvgDistUnMatched(atomPositions1, atomPositions2):

    sortedAtomPositions2 = MatchVectorOrder(atomPositions1, atomPositions2)
    return(CalcAvgDist(atomPositions1, sortedAtomPositions2))

#----------------------------------------------------------------------
# calculate the average and max distance between the corresponding
# positions in each array.
#----------------------------------------------------------------------
def CalcAvgDist(atomPositions1, atomPositions2):
    check = np.linalg.norm(atomPositions1 - atomPositions2)
    sum = 0
    maxval = 0
    for i in range(len(atomPositions1)):
        dist = np.linalg.norm(atomPositions1[i] - atomPositions2[i])
        # debugging
        # this check is correct. 'check2' concurs with 'dist' above
        #ap1 = atomPositions1[i]
        #ap2 = atomPositions2[i]
        #t1 = ap1[0] - ap2[0]
        #t2 = ap1[1] - ap2[1]
        #t3 = ap1[2] - ap2[2]
        #check2 = np.sqrt(t1*t1 + t2*t2 + t3*t3)
        sum = sum + dist
        maxval = max(dist, maxval)
    avg = sum / float(len(atomPositions1))
    return((avg, maxval))

#----------------------------------------------------------------------
# get the default center position for the residues
#----------------------------------------------------------------------
def CalcCenter(atomPositions):
    if (AVERAGE_POSITION):
        return(CalcCenter1(atomPositions))
    else:
        return(CalcCenter2(atomPositions))

#----------------------------------------------------------------------
# calculate the closest distance between non-bonded atoms
#----------------------------------------------------------------------
def CalcNearestNonBonded(atomPositions, Connections):
    min = 999.9
    # for each atom pair
    for i in range(atomPositions.shape[0]):
        for j in range(i+1, atomPositions.shape[0]):
            # if the atom pairs are non-bonded
            if (Connections.areConnected(i, j)): continue
            # calculate distance between the atom pair
            dist = np.linalg.norm(atomPositions[i] - atomPositions[j])
            # save the lowest
            if dist < min: min = dist
    return(min)

#----------------------------------------------------------------------
# calc min, avg, and max bond distances for atomPositions
# the bonds are in Connections
#----------------------------------------------------------------------
def CalcBondDistStats(atomPositions, Connections):
    sum = 0.0
    min =  999.9
    max = -999.9
    numBonds = 0
    # for each atom in atomPositions
    for i in range(atomPositions.shape[0]):
        # for each of the atom's connections
        oneAtomsConnections = Connections.m_connections[i]
        for j in range(len(oneAtomsConnections)):
            # calculate distance between these connected atoms
            dist = np.linalg.norm(atomPositions[i] - atomPositions[oneAtomsConnections[j]])
            sum += dist
            numBonds += 1
            if dist < min: min = dist
            if dist > max: max = dist
    avg = sum / float(numBonds)
    return((min, avg, max))

#----------------------------------------------------------------------
# calculate radius of gyration of the atom positions
#----------------------------------------------------------------------
def CalcRadiusOfGyration(atomPositions):
    Center = CalcCenter1(atomPositions)
    Vectors = atomPositions - Center
    NumAtoms = atomPositions.shape[0]
    RadiusOfGyration = np.linalg.norm(Vectors) / np.sqrt(float(NumAtoms))
    return(RadiusOfGyration)

#----------------------------------------------------------------------
# calculate the average of the atom positions
#----------------------------------------------------------------------
def CalcCenter1(atomPositions):
    Center = np.zeros(shape=[3], dtype=float32)
    for i in range(len(atomPositions)):
        Center = Center + atomPositions[i]
    Center = Center / len(atomPositions)
    return(Center)

#----------------------------------------------------------------------
# curMinsAndMaxs are the lowest and highest values so far
# newMinsAndMaxs are a new set of mins and maxs
# update curMinsAndMaxs with the new ones.
# if there's an update, write a message.
#----------------------------------------------------------------------
def TrackMinsAndMaxs(iter, curMinsAndMaxs, newMinsAndMaxs):

    xMin1 = curMinsAndMaxs[0]
    xMax1 = curMinsAndMaxs[1]
    yMin1 = curMinsAndMaxs[2]
    yMax1 = curMinsAndMaxs[3]
    zMin1 = curMinsAndMaxs[4]
    zMax1 = curMinsAndMaxs[5]

    xMin2 = newMinsAndMaxs[0]
    xMax2 = newMinsAndMaxs[1]
    yMin2 = newMinsAndMaxs[2]
    yMax2 = newMinsAndMaxs[3]
    zMin2 = newMinsAndMaxs[4]
    zMax2 = newMinsAndMaxs[5]

    xMin3 = min(xMin1, xMin2)
    xMax3 = max(xMax1, xMax2)
    yMin3 = min(yMin1, yMin2)
    yMax3 = max(yMax1, yMax2)
    zMin3 = min(zMin1, zMin2)
    zMax3 = max(zMax1, zMax2)

    Write = False
    if xMin3 < xMin1: Write = True
    if xMax3 > xMax1: Write = True
    if yMin3 < yMin1: Write = True
    if yMax3 > yMax1: Write = True
    if zMin3 < zMin1: Write = True
    if zMax3 > zMax1: Write = True

    if Write:
        print("%d: xMin-xMax, yMin-yMax, zMin-zMax: %8.4f -%8.4f, %8.4f -%8.4f, %8.4f -%8.4f" % (iter, xMin3, xMax3, yMin3, yMax3, zMin3, zMax3))

    return((xMin3, xMax3, yMin3, yMax3, zMin3, zMax3))

#----------------------------------------------------------------------
# rather than calculate the average of the atom positions,
# find the min & max on each axis, and take the center point.
#----------------------------------------------------------------------
def CalcCenter2(atomPositions):

    xMin = yMin = zMin = 999
    xMax = yMax = zMax = -999
    Center = np.zeros(shape=[3], dtype=float32)

    for i in range(len(atomPositions)):
        if atomPositions[i][0] > xMax:  xMax = atomPositions[i][0]
        if atomPositions[i][1] > yMax:  yMax = atomPositions[i][1]
        if atomPositions[i][2] > zMax:  zMax = atomPositions[i][2]
        if atomPositions[i][0] < xMin:  xMin = atomPositions[i][0]
        if atomPositions[i][1] < yMin:  yMin = atomPositions[i][1]
        if atomPositions[i][2] < zMin:  zMin = atomPositions[i][2]
    Center[0] = (xMin + xMax) / 2.0
    Center[1] = (yMin + yMax) / 2.0
    Center[2] = (zMin + zMax) / 2.0
    return (Center)

#----------------------------------------------------------------------
# center the atoms
#----------------------------------------------------------------------
def MoveToCenter(atomPositions, AveragePosition):
    if (AveragePosition):
        Center = CalcCenter1(atomPositions)
    else:
        Center = CalcCenter2(atomPositions)
    atomPositions = atomPositions - Center
    return(atomPositions)

#----------------------------------------------------------------------
# calculate the RMSD between 2 protein conformations.
#
# NB: For this, N is the number of atoms, whereas in the keras
# mean_square_error calculation, N is (3 * the number of atoms)
#----------------------------------------------------------------------
def CalcRMSD(atomPositions1, atomPositions2):
    N = float32(len(atomPositions1))
    dist = CalcL2Norm(atomPositions1, atomPositions2)
    return(dist / np.sqrt(N))

#----------------------------------------------------------------------
# calculate L2 norm between 2 sets of protein coordinates
#----------------------------------------------------------------------
def CalcL2Norm(atomPositions1, atomPositions2):
    assert(len(atomPositions1) == len(atomPositions2))
    dist = np.linalg.norm(np.array(atomPositions2) - np.array(atomPositions1))
    return(dist)

#----------------------------------------------------------------------
# make a PDB file from CAlphas, SideChains, and ResTypes
#----------------------------------------------------------------------
def MakePDB(AtomTypes, ResTypes, ResNums, AtomPositions, FileName):
    OutFile = open(FileName, 'w')
    OutFile.write("COMPND " + FileName + '\n')
    for i in range(len(AtomTypes)):
        OutStr = "ATOM   %4d  %1s   %3s  %4d" % (i+1, AtomTypes[i], ResTypes[i], ResNums[i])
        OutStr = OutStr + "    %8.3f%8.3f%8.3f\n" % (AtomPositions[i][0], AtomPositions[i][1], AtomPositions[i][2])
        OutFile.write(OutStr)
    OutFile.close()

#----------------------------------------------------------------------
# make a PDB file that uses InFile as a template, and replaces its
# coordinates with those in AtomPositions in the same order.
#----------------------------------------------------------------------
def MakeSimilarPDB(InFile, OutFile, AtomPositions):
    output = open(OutFile, 'w')
    input = open(InFile, 'r')
    i = 0
    for line in input:
        if line[0:4] == "ATOM":
            copyStr = line[0:30]
            positionStr = "%8.3f%8.3f%8.3f\n" % (AtomPositions[i][0], AtomPositions[i][1], AtomPositions[i][2])
            output.write(copyStr + positionStr)
            i = i + 1
        else:
            output.write(line)
    input.close()
    output.close()

#----------------------------------------------------------------------
# get 3 angles: -PI < angle < PI
# this is for doing random protein rotations
#----------------------------------------------------------------------
def Get3RandomAngles():
    Angle1 = random.uniform(-PI, PI)
    Angle2 = random.uniform(-PI, PI)
    Angle3 = random.uniform(-PI, PI)
    return((Angle1, Angle2, Angle3))

#-------------------------------------------------------------------------------
# adjust atomPositionsStop such that the new stop postions are along the same
# trajectories as before, but have compressed or expanded motions.
#
# such that:
#   stop = start + (stop - start) * noise
# where:
#   noise is in the range [TrajectoryNoiseMin**2 : TrajectoryNoiseMax**2]
#-------------------------------------------------------------------------------
def GetNoisyDeltas(atomPositionsStart, atomPositionsStop, TrajectoryNoiseMin, TrajectoryNoiseMax):

    NoiseMinSqr = TrajectoryNoiseMin * TrajectoryNoiseMin
    NoiseMaxSqr = TrajectoryNoiseMax * TrajectoryNoiseMax

    # original deltas
    deltas = atomPositionsStop - atomPositionsStart

    # make noisy deltas
    noise = np.random.rand(deltas.shape[0], deltas.shape[1])
    noise = noise * (NoiseMaxSqr - NoiseMinSqr) + NoiseMinSqr
    noisyDeltas = deltas * noise

    return(noisyDeltas)

#----------------------------------------------------------------------
# shrink or expand each bond of the protein (defined by atomPositions
# and Connections) by a random value between NoiseMin and NoiseMax
#----------------------------------------------------------------------
def AddBondDistNoise(atomPositions, Connections, NoiseMin, NoiseMax):

    assert(atomPositions.shape[0] == len(Connections.m_connections))

    newAtomPositions = np.copy(atomPositions)

    # for each atom in the protein
    for i in range(atomPositions.shape[0]):
        # for each atom it's connected to
        oneAtomsConnections = Connections.m_connections[i]
        for j in range(len(oneAtomsConnections)):
            # if connection-index is greater than atom-index (this avoids duplicates)
            if oneAtomsConnections[j] > i:
                # ii is the atom that i is connected to (when indexing into atomPositions)
                ii = oneAtomsConnections[j]
                # get the midpoint of the 2 connected atoms
                midpoint = (newAtomPositions[i] + newAtomPositions[ii]) / 2.0
                # get the distance between the 2 atoms
                dist1 = np.linalg.norm(newAtomPositions[i] - newAtomPositions[ii])
                # multiply dist by random num in range [NoiseMin : NoiseMax] to get new bond distance
                dist2 = dist1 * np.random.uniform(NoiseMin, NoiseMax)
                # reposition the endpoints
                newAtomPositions[i] = midpoint + (newAtomPositions[i] - midpoint) * dist2/dist1
                newAtomPositions[ii] = midpoint + (newAtomPositions[ii] - midpoint) * dist2/dist1
                # check that the distance is right
                # dist3 = np.linalg.norm(newAtomPositions[i] - newAtomPositions[ii])
    return(newAtomPositions)

#----------------------------------------------------------------------
# blow up or collapse the atom coordinates by dilation.
# each atom moves out or in radially from the center-of-mass.
#----------------------------------------------------------------------
def AddDilationNoise(AtomPositions, dilation):

    # get the center-of-mass
    Center = CalcCenter1(AtomPositions)

    # get vectors from each atom to the center of mass
    Vecs = AtomPositions - Center

    # multiply  the vectors by dilation
    Vecs *= dilation

    # calculate new atom-positions.  the center of mass plus the new vectors
    NewAtomPositions = Center + Vecs

    return(NewAtomPositions)

#----------------------------------------------------------------------
# add a small displacement in range [0, maxNoise] to each
# atom position in AtomPositions
#----------------------------------------------------------------------
def AddDisplacementNoise(AtomPositions, maxNoise):

    # make an array of points, where each point is a random (x,y,z) and each x/y/z is in the range [-1,1]
    rand_points = np.random.rand(AtomPositions.shape[0], AtomPositions.shape[1])
    rand_points = rand_points * 2.0 - 1.0

    # convert each (x,y,z) to a unit vector
    norms = np.linalg.norm(rand_points, axis=1).reshape(rand_points.shape[0], 1)
    rand_points = rand_points / norms

    # multiply each unit vector by a value in the range [0:maxNoise]
    mags = np.random.rand(AtomPositions.shape[0]).reshape(rand_points.shape[0], 1)
    mags = mags * maxNoise
    rand_points = rand_points * mags

    # add the random noise to AtomPositions
    AtomPositions += rand_points

    return(AtomPositions)

#----------------------------------------------------------------------
# rotate the CAlphas and SideChains Angle1, Angle2, and Angle3
# about the x, y, and z axes respectively
#----------------------------------------------------------------------
def RotateProtein(AtomPositions, Angle1, Angle2, Angle3):

    # rotate about these axes
    Axis_x = np.array([1, 0, 0], dtype=float32)
    Axis_y = np.array([0, 1, 0], dtype=float32)
    Axis_z = np.array([0, 0, 1], dtype=float32)

    # make the rotation matrices
    Matrix_x = ARotationMatrix(Axis_x, Angle1)
    Matrix_y = ARotationMatrix(Axis_y, Angle2)
    Matrix_z = ARotationMatrix(Axis_z, Angle3)

    # do the rotations
    for i in range(len(AtomPositions)):
        rotatedPos = np.matmul(Matrix_x.m_matrix, AtomPositions[i])
        rotatedPos = np.matmul(Matrix_y.m_matrix, rotatedPos)
        rotatedPos = np.matmul(Matrix_z.m_matrix, rotatedPos)
        AtomPositions[i] = rotatedPos

    # return the rotated coordinates
    return(AtomPositions)

#-------------------------------------------------------------------------------
# variation on determining the fraction of a pixel that is
# occupied by the sphere at AtomPos.
#
# use a pre-determined grid of points to calculate the fraction of points
# inside the pixel. the occupancy is the number of points inside the sphere
# divided by the total number of points in the pixel
#
# pass:
#   AtomPos - (x,y,z) tuple of an atom center.
#           - (radius of atom is AtomRadius).
#   PixelPos000 - (x,y,z) tuple of the (0,0,0) corner of a pixel.
#               - (pixel length in each dimension is PIXEL_LEN)
#
# return:
#   the fraction of the pixel that is occupied by the atom.
#-------------------------------------------------------------------------------
def CalcPixelOccupancyUsingPtsOnGrid(AtomPos, PixelPos000, AtomRadiusSqr):

    # DON'T USE THIS. Something's not right.
    # For speed: check 8 corner pixels to see if we can return 0.
    # Pts = np.add(MINIMAL_GRID_PTS, PixelPos000)
    # DistSqrs = np.sum(np.square(Pts - AtomPos), axis=1)
    # HitCount = (DistSqrs < AtomRadiusSqr).sum()
    # if (HitCount == 0):
    #     return(0.0)

    # get NUM_POINTS random (x,y,z) points, scale them to be inside PixelPos000
    # multiplication by PIXEL_LEN is done when PTS array is made
    #Pts = np.add(np.multiply(GRID_PTS, PIXEL_LEN), PixelPos000)
    Pts = np.add(GRID_PTS, PixelPos000)
    NumPoints = len(GRID_PTS)

    # get the square-of-the-distance between AtomPos, and each of the(np.random.random()*2-1) * SMALL_NUM points inside PixelPos000
    DistSqrs = np.sum(np.square(Pts - AtomPos), axis=1)

    HitCount = (DistSqrs < AtomRadiusSqr).sum()

    if (HitCount == 0):
        return(0.0)
    elif (HitCount == NumPoints):
        return(1.0)
    else:
        # add a small number (-SMALL_NUM : SMALL_NUM) to break potential ties
        tieBreaker = (np.random.random()*2-1) * (SMALL_NUM * 100)
        return(float(HitCount) / float(NumPoints) + tieBreaker)

#-------------------------------------------------------------------------------
# variation on determining the fraction of a pixel that is
# occupied by the sphere at AtomPos.
#
# scatter a bunch of random points inside the pixel. the occupancy is the
# number of points inside the sphere divided by the total number of points.
#
# pass:
#   AtomPos - (x,y,z) tuple of an atom center.
#           - (radius of atom is AtomRadius).
#   PixelPos000 - (x,y,z) tuple of the (0,0,0) corner of a pixel.
#               - (pixel length in each dimension is PIXEL_LEN)
#
# return:
#   the fraction of the pixel that is occupied by the atom.
#-------------------------------------------------------------------------------
def CalcPixelOccupancyUsingRandomNumbers(AtomPos, PixelPos000, NumPoints, AtomRadiusSqr):

    HitCount = 0

    # get NUM_POINTS random (x,y,z) points, scale them to be inside PixelPos000
    Pts = np.random.random((NumPoints, 3))
    Pts = np.add(np.multiply(Pts, PIXEL_LEN), PixelPos000)

    # get the square-of-the-distance between AtomPos, and each of the points inside PixelPos000
    DistSqrs = np.sum(np.square(Pts - AtomPos), axis=1)

    # count up the number inside the sphere
    for i in range(NumPoints):
        if DistSqrs[i] < AtomRadiusSqr:
            HitCount = HitCount+1

    return(float(HitCount)/float(NumPoints))

#-------------------------------------------------------------------------------
# rewrite of CalcPixelOccupancy. no recursion - it seems to make the code slow.
#
# pass:
#   AtomPos - (x,y,z) tuple of an atom center.
#           - (radius of atom is AtomRadius).
#   PixelPos000 - (x,y,z) tuple of the (0,0,0) corner of a pixel.
#               - (pixel length in each dimension is PIXEL_LEN)
#   the equivalent to recursion depth is hard-coded.
#
# return:
#   the fraction of the pixel that is occupied by the atom.
#   NumCubes - the number of cubes used in summing the volume.
#-------------------------------------------------------------------------------
def CalcPixelOccupancyNoRecursion(AtomPos, PixelPos000, NumCubes, AtomRadiusSqr):

    (px0, py0, pz0) = PixelPos000
    PL0 = PIXEL_LEN
    # calculate the occupied volume of 8 equal sub-pixels
    TotalVol = 0
    NumCubes[0] = 0

    #----------------
    # level 1
    #----------------
    # divide pixel into 8 equal pixels
    PL1 = PL0/2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                (px1,py1,pz1) = (px0+i*PL1, py0+j*PL1, pz0+k*PL1)
                RetVal = PixelCompletelyOccupiedOrEmpty(AtomPos, (px1,py1,pz1), PL1, AtomRadiusSqr)
                if (RetVal == 1):
                    TotalVol = TotalVol + PIXEL_VOLUMES[1]
                    #NumCubes[0] = NumCubes[0] + 1
                    continue
                elif (RetVal == 2):
                    #NumCubes[0] = NumCubes[0] + 1
                    continue

                #-----------------
                # level 2
                #-----------------
                # divide pixel into 8 equal pixels
                PL2 = PL1/2
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            (px2,py2,pz2) = (px1+ii*PL2, py1+jj*PL2, pz1+kk*PL2)
                            RetVal = PixelCompletelyOccupiedOrEmpty(AtomPos, (px2,py2,pz2), PL2, AtomRadiusSqr)
                            if (RetVal == 1):
                                TotalVol = TotalVol + PIXEL_VOLUMES[2]
                                #NumCubes[0] = NumCubes[0] + 1
                                continue
                            elif (RetVal == 2):
                                #NumCubes[0] = NumCubes[0] + 1
                                continue

                            #-----------------
                            # level 3
                            #-----------------
                            # divide pixel into 8 equal pixels
                            PL3 = PL2 / 2
                            for iii in range(2):
                                for jjj in range(2):
                                    for kkk in range(2):
                                        (px3, py3, pz3) = (px2+iii*PL3, py2+jjj*PL3, pz2+kkk*PL3)
                                        RetVal = PixelCompletelyOccupiedOrEmpty(AtomPos, (px3,py3,pz3), PL3, AtomRadiusSqr)
                                        if (RetVal == 1):
                                            TotalVol = TotalVol + PIXEL_VOLUMES[3]
                                            #NumCubes[0] = NumCubes[0] + 1
                                            continue
                                        elif (RetVal == 2):
                                            #NumCubes[0] = NumCubes[0] + 1
                                            continue

                                        TotalVol = TotalVol + PIXEL_VOLUMES_HALF[3]
                                        #NumCubes[0] = NumCubes[0] + 1

    # return the fraction of the pixel that we found was occupied
    return(TotalVol/PIXEL_VOLUMES[0])

#-------------------------------------------------------------------------------
# I'm combining PixelCompletelyOccupied and PixelCompletelyEmpty for speed.
# Some of the overhead for these is duplicated.
#
# return int:
#     0 - neither
#     1 - completely occupied
#     2 - completely empty
#-------------------------------------------------------------------------------
def PixelCompletelyOccupiedOrEmpty(AtomPos, PixelPos000, PixelLen, AtomRadiusSqr):
    (ax,  ay,  az) = AtomPos
    (px,  py,  pz) = PixelPos000
    (pxp, pyp, pzp) = np.add(PixelPos000, PixelLen)
    (ax_px,  ay_py,  az_pz)  = np.subtract(AtomPos, PixelPos000)
    (ax_pxp, ay_pyp, az_pzp) = np.subtract(AtomPos, (pxp,pyp,pzp))
    abs_ax_px = abs(ax_px)
    abs_ay_py = abs(ay_py)
    abs_az_pz = abs(az_pz)
    abs_ax_pxp = abs(ax_pxp)
    abs_ay_pyp = abs(ay_pyp)
    abs_az_pzp = abs(az_pzp)

    # find the pixel corner farthest from AtomPos. (it goes in (x,y,z))
    (x,y,z) = (px, py, pz)
    if abs_ax_px < abs_ax_pxp : x = pxp
    if abs_ay_py < abs_ay_pyp : y = pyp
    if abs_az_pz < abs_az_pzp : z = pzp

    # if the farthest corner is inside the atom's radius, the full pixel is inside the atom's radius
    (ax_x, ay_y, az_z) = np.subtract((ax,ay,az), (x,y,z))
    distSqr = ax_x*ax_x + ay_y*ay_y + az_z*az_z
    if distSqr < AtomRadiusSqr:
        return(1)

    # find the pixel corner closest to AtomPos. (it goes in (x,y,z))
    (x,y,z) = (px, py, pz)
    if abs_ax_px > abs_ax_pxp : x = pxp
    if abs_ay_py > abs_ay_pyp : y = pyp
    if abs_az_pz > abs_az_pzp : z = pzp

    # if the closest corner is outside the atom's radius by a sufficient margin,
    # then the full pixel is outside the atom's radius. see my 2018 notebook, p. 104.
    (ax_x, ay_y, az_z) = np.subtract((ax,ay,az), (x,y,z))
    distSqr = ax_x*ax_x + ay_y*ay_y + az_z*az_z
    if distSqr > AtomRadiusSqr + PixelLen*PixelLen/2.0:
        return(2)

    # the pixel is partially occupied
    return(0)

#-------------------------------------------------------------------------------
# check if the pixel whose (0,0,0) corner is PixelPos000 is completely
# inside the radius of the sphere at AtomPos.
#-------------------------------------------------------------------------------
def PixelCompletelyOccupied(AtomPos, PixelPos000, PixelLen, AtomRadiusSqr):
    (ax, ay, az) = AtomPos
    (px, py, pz) = PixelPos000
    (pxp, pyp, pzp) = (px+PixelLen, py+PixelLen, pz+PixelLen)

    # find the pixel corner farthest from AtomPos. (it goes in (x,y,z))
    (x,y,z) = (px, py, pz)
    if abs(ax - px) < abs(ax - pxp) : x = pxp
    if abs(ay - py) < abs(ay - pyp) : y = pyp
    if abs(az - pz) < abs(az - pzp) : z = pzp

    # if the farthest corner is inside the atom's radius, the full pixel is inside the atom's radius
    axx = ax-x
    ayy = ay-y
    azz = az-z
    distSqr = axx*axx + ayy*ayy + azz*azz
    if distSqr < AtomRadiusSqr:
        return(True)
    return(False)

#-------------------------------------------------------------------------------
# check if the pixel whose (0,0,0) corner is PixelPos000 is completely
# outside the radius of the sphere at AtomPos.
#-------------------------------------------------------------------------------
def PixelCompletelyEmpty(AtomPos, PixelPos000, PixelLen, AtomRadiusSqr):
    (ax, ay, az) = AtomPos
    (px, py, pz) = PixelPos000
    (pxp, pyp, pzp) = (px+PixelLen, py+PixelLen, pz+PixelLen)

    # find the pixel corner closest to AtomPos. (it goes in (x,y,z))
    (x,y,z) = (px, py, pz)
    if abs(ax - px) > abs(ax - pxp) : x = pxp
    if abs(ay - py) > abs(ay - pyp) : y = pyp
    if abs(az - pz) > abs(az - pzp) : z = pzp

    # if the closest corner is outside the atom's radius by a sufficient margin,
    # then the full pixel is outside the atom's radius. see my 2018 notebook, p. 104.
    axx = ax-x
    ayy = ay-y
    azz = az-z
    distSqr = axx*axx + ayy*ayy + azz*azz
    if distSqr > AtomRadiusSqr + PixelLen*PixelLen/2.0:
        return(True)
    return(False)

#-------------------------------------------------------------------------------
# pass:
#   AtomPos - (x,y,z) tuple of an atom center.
#           - (radius of atom is AtomRadius).
#   PixelPos000 - (x,y,z) tuple of the (0,0,0) corner of a pixel.
#               - (pixel length in each dimension is PIXEL_LEN)
#   MaxLevel - maximum number of recursions.
#              1 => 8 cubes
#              2 => 64 cubes
#              3 => 512 cubes ...
#
# return:
#   the fraction of the pixel that is occupied by the atom.
#   NumCubes - the number of cubes used in summing the volume.
#-------------------------------------------------------------------------------
def CalcPixelOccupancy(AtomPos, PixelPos000, MaxLevel, NumCubes, AtomRadiusSqr):
    (px, py, pz) = PixelPos000
    PL2 = PIXEL_LEN/2
    # calculate the occupied volume of 8 equal sub-pixels
    TotalVol = 0
    NumCubes[0] = 0
    TotalVol += CalcSubPixelOccupancy(AtomPos, PL2, (px,     py,     pz    ), 1, MaxLevel, NumCubes, AtomRadiusSqr)
    TotalVol += CalcSubPixelOccupancy(AtomPos, PL2, (px,     py,     pz+PL2), 1, MaxLevel, NumCubes, AtomRadiusSqr)
    TotalVol += CalcSubPixelOccupancy(AtomPos, PL2, (px,     py+PL2, pz    ), 1, MaxLevel, NumCubes, AtomRadiusSqr)
    TotalVol += CalcSubPixelOccupancy(AtomPos, PL2, (px,     py+PL2, pz+PL2), 1, MaxLevel, NumCubes, AtomRadiusSqr)
    TotalVol += CalcSubPixelOccupancy(AtomPos, PL2, (px+PL2, py,     pz    ), 1, MaxLevel, NumCubes, AtomRadiusSqr)
    TotalVol += CalcSubPixelOccupancy(AtomPos, PL2, (px+PL2, py,     pz+PL2), 1, MaxLevel, NumCubes, AtomRadiusSqr)
    TotalVol += CalcSubPixelOccupancy(AtomPos, PL2, (px+PL2, py+PL2, pz    ), 1, MaxLevel, NumCubes, AtomRadiusSqr)
    TotalVol += CalcSubPixelOccupancy(AtomPos, PL2, (px+PL2, py+PL2, pz+PL2), 1, MaxLevel, NumCubes, AtomRadiusSqr)
    return(TotalVol/(PIXEL_VOLUMES[0]))

#-------------------------------------------------------------------------------
# pass:
#   AtomPos - (x,y,z) tuple of an atom center. (radius of atom is AtomRadius).
#   PixelLen - the length of each pixel dimension.
#   PixelPos000 - (x,y,z) tuple of the (0,0,0) corner of a pixel.
#   Level - the recursion level.
#
# return:
#   - if the pixel is fully in the atom radius, return the pixel volume.
#   - if the pixel is fully outside the atom radius, return 0.
#   - if the pixel is partially occupied, divide the pixel in 8 pieces,
#     and call this routine at a higher recursion level.
#   - if the highest recursion level is reached, check the midpoint
#     of the pixel. if it's in the sphere, return the pixel volume,
#     if it's outside the sphere return 0.
#   NumCubes - the number of cubes used in summing the volume.
#
# For the algorithm, see my 2018 notebook, p. 102.
#-------------------------------------------------------------------------------
def CalcSubPixelOccupancy(AtomPos, PixelLen, PixelPos000, CurrentLevel, MaxLevel, NumCubes, AtomRadiusSqr):

    #pt1 = np.array(AtomPos)  # the atom center
    (ax, ay, az) = AtomPos
    (px, py, pz) = PixelPos000
    (pxp, pyp, pzp) = (PixelPos000[0]+PixelLen, PixelPos000[1]+PixelLen, PixelPos000[2]+PixelLen)

    #---------------------------------------------------
    # if the highest level of recursion is reached
    #---------------------------------------------------
    if CurrentLevel == MaxLevel:
        # see my 2018 notebook, p. 110.
        # add 1/2 this cube's volume to the volume sum.
        NumCubes[0] = NumCubes[0] + 1
        return(PIXEL_VOLUMES[CurrentLevel]/2)

    #---------------------------------------------------
    # test the pixel for 100% occupancy:
    #---------------------------------------------------
    # find the pixel corner farthest from AtomPos. (it goes in (x,y,z))
    (x,y,z) = (px, py, pz)
    if abs(ax - px) < abs(ax - pxp) : x = pxp
    if abs(ay - py) < abs(ay - pyp) : y = pyp
    if abs(az - pz) < abs(az - pzp) : z = pzp
    # if the farthest corner is inside the atom's radius, the full pixel is inside the atom's radius
    #pt2 = np.array((x,y,z))  # the corner of the pixel farthest from the atom center
    #dist = np.linalg.norm(pt1 - pt2)
    axx = ax-x
    ayy = ay-y
    azz = az-z
    distSqr = axx*axx + ayy*ayy + azz*azz
    if distSqr < AtomRadiusSqr:
        NumCubes[0] = NumCubes[0] + 1
        return(PIXEL_VOLUMES[CurrentLevel])

    #---------------------------------------------------
    # test the pixel for 0% occupancy:
    #---------------------------------------------------
    # find the pixel corner closest to AtomPos. (it goes in (x,y,z))
    (x,y,z) = (px, py, pz)
    if abs(ax - px) > abs(ax - pxp) : x = pxp
    if abs(ay - py) > abs(ay - pyp) : y = pyp
    if abs(az - pz) > abs(az - pzp) : z = pzp
    # if the closest corner is outside the atom's radius by a sufficient margin,
    # then the full pixel is outside the atom's radius. see my 2018 notebook, p. 104.
    #pt2 = np.array((x,y,z))  # the corner of the pixel farthest from the atom center
    #dist = np.linalg.norm(pt1 - pt2)
    axx = ax-x
    ayy = ay-y
    azz = az-z
    distSqr = axx*axx + ayy*ayy + azz*azz
    #if dist > np.sqrt(AtomRadiusSqr + PIXEL_LENS_SQR[CurrentLevel]/2.0):
    if distSqr > AtomRadiusSqr + PIXEL_LENS_SQR[CurrentLevel]/2.0:
        NumCubes[0] = NumCubes[0] + 1
        return(0)

    #------------------------------------------------------------
    # divide the pixel into 8 cubes and call this same routine
    # do this when the pixel is neither 100% or 0% occupied
    #------------------------------------------------------------
    (px, py, pz) = PixelPos000
    PL2 = PixelLen/2
    # calculate the occupied volume of 8 equal sub-pixels
    Total = 0
    Total += CalcSubPixelOccupancy(AtomPos, PL2, (px,     py,     pz    ), CurrentLevel+1, MaxLevel, NumCubes, AtomRadiusSqr)
    Total += CalcSubPixelOccupancy(AtomPos, PL2, (px,     py,     pz+PL2), CurrentLevel+1, MaxLevel, NumCubes, AtomRadiusSqr)
    Total += CalcSubPixelOccupancy(AtomPos, PL2, (px,     py+PL2, pz    ), CurrentLevel+1, MaxLevel, NumCubes, AtomRadiusSqr)
    Total += CalcSubPixelOccupancy(AtomPos, PL2, (px,     py+PL2, pz+PL2), CurrentLevel+1, MaxLevel, NumCubes, AtomRadiusSqr)
    Total += CalcSubPixelOccupancy(AtomPos, PL2, (px+PL2, py,     pz    ), CurrentLevel+1, MaxLevel, NumCubes, AtomRadiusSqr)
    Total += CalcSubPixelOccupancy(AtomPos, PL2, (px+PL2, py,     pz+PL2), CurrentLevel+1, MaxLevel, NumCubes, AtomRadiusSqr)
    Total += CalcSubPixelOccupancy(AtomPos, PL2, (px+PL2, py+PL2, pz    ), CurrentLevel+1, MaxLevel, NumCubes, AtomRadiusSqr)
    Total += CalcSubPixelOccupancy(AtomPos, PL2, (px+PL2, py+PL2, pz+PL2), CurrentLevel+1, MaxLevel, NumCubes, AtomRadiusSqr)
    return(Total)

#-------------------------------------------------------------------------------
# print atomPositions
# Indices and Distances have optional supplementary info
#-------------------------------------------------------------------------------
def printAtomPositions(atomPositions, label, Indices, Dists):
    print(label)
    for i in range(len(atomPositions)):
        Pos = atomPositions[i]
        if Indices is None:
            print("%2d: x, y, z = %7.3f, %7.3f, %7.3f" % (i, Pos[0], Pos[1], Pos[2]))
        else:
            print("%2d: x, y, z = %7.3f, %7.3f, %7.3f  ->  %2d  %5.3f" % (i, Pos[0], Pos[1], Pos[2], Indices[i], Dists[i]))

#-------------------------------------------------------------------------------
# for each Position1, find the Position2 its closest to
# return the indices, and the distances
#-------------------------------------------------------------------------------
def getClosestPoints(Positions1, Positions2):
    Indices = []
    Distances = []
    AvgDist = 999
    for Pos1 in Positions1:
        min = 999
        index = 0
        saveindex = 0
        for Pos2 in Positions2:
            dist = np.linalg.norm(Pos1 - Pos2)
            if dist < min:
                min = dist
                saveindex = index
            index += 1
        Indices.append(saveindex)
        Distances.append(min)
        Dists = np.array(Distances)
        Sum = np.sum(Dists)
        AvgDist = Sum / len(Dists)
    return(Indices, Distances, AvgDist)

#-------------------------------------------------------------------------------
# print the closest matches between atomPositions and estimatedPositions
# only print a match if the closest-match is reciprocal
# return the distances between matches
#-------------------------------------------------------------------------------
def printMatchingPositions(atomPositions, estimatedPositions):

    distances = []

    # for each true atom position, find the estimated position it's closest to
    (closestIndices1, closestDistances1, avgDist1) = getClosestPoints(atomPositions, estimatedPositions)

    # if the avg dist is too big, flip the signs on the estimated positions and try again
    if avgDist1 > 1.0:
        estimatedPositions = np.array(estimatedPositions) * -1.0
        (closestIndices1, closestDistances1, avgDist1) = getClosestPoints(atomPositions, estimatedPositions)

    # for each estimated position, find the true atom position its closest to
    (closestIndices2, closestDistances2, avgDist2) = getClosestPoints(estimatedPositions, atomPositions)

    # for each atom position
    i = 0
    for aPos in atomPositions:
        # write the atom position
        print("%2d: true : estimated: (%7.3f, %7.3f, %7.3f) : " % (i, aPos[0], aPos[1], aPos[2]), end='')
        # get the closest estimated position
        ePos = estimatedPositions[closestIndices1[i]]
        # if this closest atom points back to atom
        index = closestIndices2[closestIndices1[i]]
        if index == i:
            # print it
            print("(%7.3f, %7.3f, %7.3f) = %5.3f" % (ePos[0], ePos[1], ePos[2], closestDistances1[i]))
            distances.append(closestDistances1[i])
        else:
            # otherwise, there's no unambiguous match
            print("")
        i += 1

    return(distances)

#-------------------------------------------------------------------------------
# Vals is a list of values
# put the values in bins, then print the bins
#-------------------------------------------------------------------------------
def PrintHistogram(Vals, Start, Stop, Inc):

    # make the bins
    Bins = [0] * round((Stop - Start) / Inc)

    # fill the bins
    avg = 0
    for val in Vals:
        BinIndex = int(val/Inc)
        Bins[BinIndex] += 1
        avg += val
    avg /= float(len(Vals))

    # print the results
    for i in range(len(Bins)):
        binStart = Start + i * Inc
        binStop = binStart + Inc
        print("%5.2f - %5.2f: %d" % (binStart, binStop, Bins[i]))

    print("number of items = %d" % (len(Vals)))
    print("average value = %5.3f" % (avg))

