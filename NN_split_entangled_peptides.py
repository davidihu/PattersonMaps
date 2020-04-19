# author: David Hurwitz
# started: 8/17/19
#

from NN_formatted_data import OneNNData, NUM_PIXELS_1D
from NN_misc import PIXEL_LEN
import random
import numpy as np

AtomRadiusSqr = PIXEL_LEN * PIXEL_LEN
NumAtoms = 40
MidSlice = round(NUM_PIXELS_1D/2)

# make an empty OneNNData
data = OneNNData()

for ii in range(100):

    # get NumAtoms random coordinates.  these are the true coordinates.
    atomPositions = data.GetRandomPositions2(NumAtoms, AtomRadiusSqr)
    atomTypes = ['C'] * NumAtoms

    # calculate the density map for atomPositions and put it on the NN output.
    data.MakeMap(atomPositions, atomTypes, False, AtomRadiusSqr, ClearData=True)

    # calculate the Patterson map from the NN output and put it on the NN input. this is the true Patterson map.
    # don't need to save the density map on the NN output any longer.
    data.MakePattersonMap()

    # with 50% chance, flip each position from (x,y,z) -> (-x,-y,-z)
    randomizedPositions = data.RandomizePositions(atomPositions)
    randomizedPositions = np.asarray(randomizedPositions)

    # calculate the density map for randomizedPositions and put it on the NN output. this is the test density map.
    data.MakeMap(randomizedPositions, atomTypes, False, AtomRadiusSqr, ClearData=True)

    # calculate the Patterson map from the NN output and put it on the work array. this is the test Patterson map.
    data.MakePattersonMap(outputIsWorkArray=True)

    # the original test density
    # data.PrintSlice(2, MidSlice, False)

    # get a similarity score (MSE) between the true Patterson map and the test Patterson map.
    SimilarityScore = data.ComparePattersonMaps()
    print("\nSimilarity Score = %f" %(SimilarityScore))

    # repeat many times
    count = 0
    while SimilarityScore > 0.0001:

        # choose a random position in randomizedPositions
        index = random.randrange(len(randomizedPositions))

        # subtract the density for the initial value from the NN output.
        Pos = randomizedPositions[index]
        data.AddToDensity(Pos, False, AtomRadiusSqr, SubtractInstead=True)

        # add the density for the inverted value to the NN output.
        data.AddToDensity(Pos*-1.0, False, AtomRadiusSqr)

        # calculate the Patterson map from the NN output and put it on the work array.  this is the test Patterson map.
        data.MakePattersonMap(outputIsWorkArray=True)

        # get a similarity score (mean-squared-error) between the true Patterson map and the test Patterson map.
        TestSimilarityScore = data.ComparePattersonMaps()

        # if the similarity score improved
        if TestSimilarityScore < (SimilarityScore + .01):

            # update the similarity score
            if TestSimilarityScore > SimilarityScore:
                print("Similarity Score = %f *" % (TestSimilarityScore))
            else:
                print("Similarity Score = %f" % (TestSimilarityScore))
            SimilarityScore = TestSimilarityScore

            # save the flipped signs
            randomizedPositions[index] = Pos * -1.0

            # the latest test density
            # data.PrintSlice(2, MidSlice, False)

        # if the similarity score got worse
        else:

            # subtract the density for the inverted value from the NN output.
            data.AddToDensity(Pos*-1, False, AtomRadiusSqr, SubtractInstead=True)

            # add the density for the initial value to the NN output.
            data.AddToDensity(Pos, False, AtomRadiusSqr)

        count += 1
        if count > 1000:
            break

    for i in range(len(atomPositions)):
        pos1 = atomPositions[i]
        pos2 = randomizedPositions[i]
        print("(%7.3f,%7.3f,%7.3f) (%7.3f,%7.3f,%7.3f)" % (pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]))

# code below was for testing. it's not needed.
exit()

# fully enumerate the possibilities
count = 0
goodCount = 0
for i0 in range(2):
    for i1 in range(2):
        for i2 in range(2):
            for i3 in range(2):
                for i4 in range(2):
                    for i5 in range(2):
                        for i6 in range(2):
                            for i7 in range(2):
                                for i8 in range(2):
                                    for i9 in range(2):

                                        # start from the atom positions that produced the density map
                                        testPositions = atomPositions.copy()

                                        # flip signs on some of the atom positions
                                        if i0 == 1: testPositions[0] *= -1.0
                                        if i1 == 1: testPositions[1] *= -1.0
                                        if i2 == 1: testPositions[2] *= -1.0
                                        if i3 == 1: testPositions[3] *= -1.0
                                        if i4 == 1: testPositions[4] *= -1.0
                                        if i5 == 1: testPositions[5] *= -1.0
                                        if i6 == 1: testPositions[6] *= -1.0
                                        if i7 == 1: testPositions[7] *= -1.0
                                        if i8 == 1: testPositions[8] *= -1.0
                                        if i9 == 1: testPositions[9] *= -1.0

                                        # make a density map on the output
                                        data.MakeMap(testPositions, atomTypes, False, AtomRadiusSqr, ClearData=True)

                                        # make a test Patterson map on the work array
                                        data.MakePattersonMap(outputIsWorkArray=True)

                                        # calculate a similarity score between the test Patterson map and the good one
                                        TestSimilarityScore = data.ComparePattersonMaps()

                                        # if the similiarity score is sufficiently low
                                        if (TestSimilarityScore < 0.001):
                                            print("%4d: Similarity Score = %f" % (count, SimilarityScore))
                                            goodCount += 1

                                        count += 1

print("good count = %d" % (goodCount))
test = 1