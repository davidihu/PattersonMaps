# author: David Hurwitz
# started: 7/18/18
#

from NN_raw_data import ALine, AllLines, CAlphaIndices
from NN_misc import Get3RandomAngles, RotateProtein, MakePDB, SMALL_NUM, PI
from NN_misc import AddDisplacementNoise, GetNoisyDeltas, AddDilationNoise, AddBondDistNoise
from NN_misc import STEP_NUM_INC, RES_TYPES, PIXEL_LEN, PIXEL_LEN_CUBED
from NN_misc import CalcPixelOccupancy, CalcPixelOccupancyNoRecursion
from NN_misc import CalcAvgDist, CalcAvgDistUnMatched, MatchVectorOrder
from NN_misc import CalcPixelOccupancyUsingRandomNumbers, CalcPixelOccupancyUsingPtsOnGrid
from NN_color_scales import get_color, get_color_scaled
import numpy as np
import time
from numpy import float32, fft, conj, real
from matplotlib import pyplot
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D

# BOX_MIN = -16.00    # box minimum in all 3 dimensions
# BOX_MAX =  16.00    # box maximum in all 3 dimensions
# BOX_MIN = -17.20    # box minimum in all 3 dimensions
# BOX_MAX =  17.20    # box maximum in all 3 dimensions
# BOX_MIN = -18.00    # box minimum in all 3 dimensions
# BOX_MAX =  18.00    # box maximum in all 3 dimensions
BOX_MIN = -20.00    # box minimum in all 3 dimensions
BOX_MAX =  20.00    # box maximum in all 3 dimensions
# BOX_MIN = -22.00    # box minimum in all 3 dimensions
# BOX_MAX =  22.00    # box maximum in all 3 dimensions
# BOX_MIN = -32.00    # box minimum in all 3 dimensions
# BOX_MAX =  32.00    # box maximum in all 3 dimensions

NUM_PIXELS_1D = round((BOX_MAX-BOX_MIN)/PIXEL_LEN)   # num pixels in each dimension

NUM_IN_CHANNELS = 1          # number of channels in OneNNData::InData
NUM_OUT_CHANNELS = 1         # number of channels in OneNNData::OutData
NUM_WORK_CHANNELS = 1        # number of channels in OneNNData::WorkArray

# NN formatted data for one time-step
class OneNNData:

    #--------------------------------------------------------------------------------
    # make numpy arrays for the formatted input and output mapes
    # each map is either a density map or connection map
    #--------------------------------------------------------------------------------
    def __init__(self):

        # make numpy arrays to store the formatted input and output data
        np_1d = NUM_PIXELS_1D
        self.InData = np.zeros(shape=(np_1d, np_1d, np_1d, NUM_IN_CHANNELS), dtype=float32)
        self.OutData = np.zeros(shape=(np_1d, np_1d, np_1d, NUM_OUT_CHANNELS), dtype=float32)
        self.WorkArray = np.zeros(shape=(np_1d, np_1d, np_1d, NUM_WORK_CHANNELS), dtype=float32)

        # save a record of the raw data used to make the formatted data
        Temp = []
        self.AtomTypes = []               # atom-types
        self.AtomPositions = []           # atom positions

    #--------------------------------------------------------------------------------
    # get the raw data for 1 time step
    #--------------------------------------------------------------------------------
    def GetRawData(self):
        return(self.AtomTypes, self.AtomPositions)

    #--------------------------------------------------------------------------------
    # clear the current NN data
    #--------------------------------------------------------------------------------
    def ClearNN(self):
        self.InData.fill(0.0)
        self.OutData.fill(0.0)
        self.WorkArray.fill(0.0)

    #--------------------------------------------------------------------------------
    # get max from one input or output channel
    #--------------------------------------------------------------------------------
    def GetMaxVal(self, InData, Channel=0):
        if (InData):
            return(self.InData[:,:,:,Channel].max())
        else:
            return(self.OutData[:,:,:,Channel].max())

    #--------------------------------------------------------------------------------
    # get a value from one input or output channel
    #--------------------------------------------------------------------------------
    def GetOneVal(self, i, j, k, InData, Channel=0):
        if (InData):
            return(self.InData[i, j, k, Channel])
        else:
            return(self.OutData[i, j, k, Channel])

    #--------------------------------------------------------------------------------
    # set a value from the input or output channels
    #--------------------------------------------------------------------------------
    def SetOneVal(self, i, j, k, InData, Val, Channel=0):
        if (InData):
            self.InData[i, j, k, Channel] = Val
        else:
            self.OutData[i, j, k, Channel] = Val

    #--------------------------------------------------------------------------------
    # increment a value from the input or output channels by Amt
    #--------------------------------------------------------------------------------
    def IncOneVal(self, i, j, k, InData, Amt, Channel=0):
        if (InData):
            self.InData[i, j, k, Channel] += Amt
        else:
            self.OutData[i, j, k, Channel] += Amt

    #--------------------------------------------------------------------------------
    # Print the difference between 2 images.
    # Negative diffs get different colors from positive diffs.
    #--------------------------------------------------------------------------------
    def PrintImageDiff(self, image1, image2):

        temp_image = image1 - image2
        new_image = np.zeros(shape=(NUM_PIXELS_1D, NUM_PIXELS_1D, 3), dtype=float32)

        test0 = temp_image[:,:,0].sum()
        test1 = temp_image[:,:,1].sum()
        test2 = temp_image[:,:,2].sum()

        # get min and max of (image1 - image2)
        max = np.amax(temp_image)
        min = np.amin(temp_image)
        print("min temp_image value = %f" % min)
        print("max temp_image value = %f" % max)

        # scale and color the new_image
        for i in range(NUM_PIXELS_1D):
            for j in range(NUM_PIXELS_1D):

                # red-channel pos diffs become red + green
                if temp_image[i, j, 0] > SMALL_NUM:
                    new_image[i, j, 0] = temp_image[i, j, 0] * (1.0 / max)
                    new_image[i, j, 1] = temp_image[i, j, 0] * (1.0 / max)

                # red-channel neg diffs become red + blue
                if temp_image[i, j, 0] < -SMALL_NUM:
                    new_image[i, j, 0] = temp_image[i, j, 0] * (1.0 / min)
                    new_image[i, j, 2] = temp_image[i, j, 0] * (1.0 / min)

                # green-channel pos diffs become green + blue
                if temp_image[i, j, 1] > SMALL_NUM:
                    new_image[i, j, 1] = temp_image[i, j, 1] * (1.0 / max)
                    new_image[i, j, 2] = temp_image[i, j, 1] * (1.0 / max)

                # green-channel neg diffs become green + red
                if temp_image[i, j, 1] < -SMALL_NUM:
                    new_image[i, j, 1] = temp_image[i, j, 1] * (1.0 / min)
                    new_image[i, j, 0] = temp_image[i, j, 1] * (1.0 / min)

                # blue-channel pos diffs become blue + red
                if temp_image[i, j, 2] > SMALL_NUM:
                    new_image[i, j, 2] = temp_image[i, j, 2] * (1.0 / max)
                    new_image[i, j, 0] = temp_image[i, j, 2] * (1.0 / max)

                # blue-channel neg diffs become blue + green
                if temp_image[i, j, 2] < -SMALL_NUM:
                    new_image[i, j, 2] = temp_image[i, j, 2] * (1.0 / min)
                    new_image[i, j, 1] = temp_image[i, j, 2] * (1.0 / min)

        max2 = np.amax(new_image)
        min2 = np.amin(new_image)

        # show new_image
        pyplot.figure(3)
        pyplot.imshow(new_image)
        pyplot.show()

    #--------------------------------------------------------------------------------
    # make a density map from saved atom positions
    # determine the voxels and facecolors arrays for 3d display
    # make a density map from atoms that are centro-symmetric from the saved atoms.
    # determine the voxels and facecolors arrays fro 3d display
    # merge the voxels and facecolors arrays
    # draw in 3d
    #--------------------------------------------------------------------------------
    def Print3dTest3(self, AtomRadiusSqr, Threshold, Title, FileNum, printOriginal=True, printCentroSymmetric=True, Opacity=0.5):

        path = "C:/Users/david/Documents/newff/results/NN/Patterson_maps/figures/3d_images_2/"
        Original = "Original" if printOriginal else ""
        CentroSym = "CentroSymmetric" if printCentroSymmetric else ""
        saveFile = path + "TrueOutput_%s%s_%02d.png" % (Original, CentroSym, FileNum+1)

        # these are the true atom positions
        Atoms = np.copy(self.AtomPositions)
        Types = ['C'] * len(Atoms)

        figNum = 0
        if     printOriginal and not printCentroSymmetric: figNum = 1
        if not printOriginal and     printCentroSymmetric: figNum = 2
        if     printOriginal and     printCentroSymmetric: figNum = 3

        fig = pyplot.figure(num=figNum, figsize=(10,9))
        pyplot.clf()
        ax = fig.gca(projection='3d')
        pyplot.title(Title, y=1.05)

        # calculate density map for the original atoms. cut out peripheral white space.
        self.MakeMap(Atoms, Types, False, AtomRadiusSqr, ClearData=True, SavePositions=False)
        Density1 = np.copy(self.OutData[10:30,10:30,10:30,0]) / 2

        # calculate density map for the centro-symmetric atoms. cut out peripheral white space.
        Atoms = Atoms * -1
        self.MakeMap(Atoms, Types, False, AtomRadiusSqr, ClearData=True, SavePositions=False)
        Density2 = np.copy(self.OutData[10:30,10:30,10:30,0]) / 2

        # make a combined density map for showing overlaps
        Density3 = np.add(Density1, Density2)

        # make empty color arrays, to be filled in where needed with facecolors
        colors1 = np.empty(Density1.shape, dtype=object)
        colors2 = np.empty(Density2.shape, dtype=object)
        colors3 = np.empty(Density3.shape, dtype=object)

        # save True or False to indicate where there's density for the original atoms
        voxels1 = Density1 > 999
        if printOriginal:
            voxels1 = Density1 > float(Threshold)
            colors1 = self.MakeColorsArray(Density1, voxels1, True, False, False, False, Opacity)
        print("Test3: Number set for original atoms = %4d" % (np.sum(voxels1)))

        # save True or False to indicate where there's density for the centro-symmetric atoms
        voxels2 = Density2 > 999
        if printCentroSymmetric:
            voxels2 = Density2 > float(Threshold)
            colors2 = self.MakeColorsArray(Density2, voxels2, False, False, True, False, Opacity)
        print("Test3: Number set for centro-symmetric atoms = %4d" % (np.sum(voxels2)))

        # make a combined True/False array
        voxels_or = np.logical_or(voxels1, voxels2)
        print("Test3: Number set for all atoms = %4d" % (np.sum(voxels_or)))

        # make an overlap True/False array
        voxels_and = np.logical_and(voxels1, voxels2)
        print("Test3: Number overlapping = %4d" % (np.sum(voxels_and)))

        # make a colors array for the overlapping pixels
        if printOriginal and printCentroSymmetric:
            colors3 = self.MakeColorsArray(Density3, voxels_and, False, True, False, False, Opacity)

        # make a combined colors array
        colors = np.empty(voxels1.shape, dtype=object)
        for i in range(colors.shape[0]):
            for j in range(colors.shape[1]):
                for k in range(colors.shape[2]):
                    color1Set = colors1[i,j,k] is not None
                    color2Set = colors2[i,j,k] is not None
                    if (not color1Set) and (not color2Set):  continue                           # no color set
                    if (    color1Set) and (not color2Set):  colors[i,j,k] = colors1[i,j,k]     # red
                    if (not color1Set) and (    color2Set):  colors[i,j,k] = colors2[i,j,k]     # blue
                    if (    color1Set) and (    color2Set):  colors[i,j,k] = colors3[i,j,k]     # purple

        ax.voxels(voxels_or, facecolors=colors, edgecolors=(0.5, 0.5, 0.5, Opacity))

        # make the legend
        handles = ()
        spacer = patches.Patch(color='white', label=None)
        if printOriginal:
            red1 = patches.Patch(color=get_color_scaled(0.0,  True, False, False, False), label='0.00')
            red2 = patches.Patch(color=get_color_scaled(0.25, True, False, False, False), label='0.25')
            red3 = patches.Patch(color=get_color_scaled(0.5,  True, False, False, False), label='0.50')
            if printCentroSymmetric:
                handles = (red1, red2, red3, spacer)
            else:
                handles = (red1, red2, red3)
        if printCentroSymmetric:
            blue1 = patches.Patch(color=get_color_scaled(0.0,  False, False, True, False), label='0.00')
            blue2 = patches.Patch(color=get_color_scaled(0.25, False, False, True, False), label='0.25')
            blue3 = patches.Patch(color=get_color_scaled(0.5,  False, False, True, False), label='0.50')
            if printOriginal:
                handles = handles + (blue1, blue2, blue3, spacer)
            else:
                handles = handles + (blue1, blue2, blue3)
        if printOriginal and printCentroSymmetric:
            green1 = patches.Patch(color=get_color_scaled(0.0,  False, True, False, False), label='0.00')
            green2 = patches.Patch(color=get_color_scaled(0.25, False, True, False, False), label='0.25')
            green3 = patches.Patch(color=get_color_scaled(0.5,  False, True, False, False), label='0.50')
            green4 = patches.Patch(color=get_color_scaled(0.75, False, True, False, False), label='0.75')
            green5 = patches.Patch(color=get_color_scaled(1.0,  False, True, False, False), label='1.00')
            handles = handles + (green1, green2, green3, green4, green5)
        pyplot.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.10, 1.00))
        time.sleep(2)
        pyplot.savefig(saveFile, format='png')

    #--------------------------------------------------------------------------------
    # make array with colors for the set voxels.
    # the bigger the Density value, the brighter the color.
    #--------------------------------------------------------------------------------
    def MakeColorsArray(self, Density, Voxels, Red, Green, Blue, Purple, Opacity):

        Colors = np.empty(Density.shape, dtype=object)
        for i in range(Colors.shape[0]):
            for j in range(Colors.shape[1]):
                for k in range(Colors.shape[2]):
                    if Voxels[i,j,k]:
                        val = float(Density[i, j, k])
                        color = get_color_scaled(val, Red, Green, Blue, Purple)
                        Colors[i, j, k] = (color[0], color[1], color[2], Opacity)
        return(Colors)

    #--------------------------------------------------------------------------------
    # draw the current NN output in 3d
    #--------------------------------------------------------------------------------
    def Print3dTest2(self, Threshold, Title, FileNum, Opacity):

        path = "C:/Users/david/Documents/newff/results/NN/Patterson_maps/figures/3d_images_2/"
        saveFile = path + "PredictedOutput_%02d.png" % (FileNum+1)

        figNum = 4
        fig = pyplot.figure(num=figNum, figsize=(10,9))
        pyplot.clf()
        ax = fig.gca(projection='3d')
        pyplot.title(Title, y=1.05)

        voxels = self.OutData[10:30,10:30,10:30,0] > float(Threshold)
        colors = np.empty(voxels.shape, dtype=object)
        for i in range(colors.shape[0]):
            for j in range(colors.shape[1]):
                for k in range(colors.shape[2]):
                    if voxels[i,j,k]:
                        val = float(self.OutData[i+10,j+10,k+10])
                        if val > 0.5:
                            color = get_color_scaled(val, False, True, False, False)
                        else:
                            color = get_color_scaled(val, False, False, False, True)
                        colors[i, j, k] = (color[0], color[1], color[2], Opacity)

        print("Test2: Number set = %4d" % (self.CountNumberSet(voxels)))

        ax.voxels(voxels, facecolors=colors, edgecolors=(0.5, 0.5, 0.5, Opacity))

        # make the legend
        handles = ()
        purple1 = patches.Patch(color=get_color_scaled(0.0,  False, False, False, True), label='0.00')
        purple2 = patches.Patch(color=get_color_scaled(0.25, False, False, False, True), label='0.25')
        purple3 = patches.Patch(color=get_color_scaled(0.5,  False, False, False, True), label='0.50')
        green1 =  patches.Patch(color=get_color_scaled(0.5,  False, True, False, False), label='0.50')
        green2 =  patches.Patch(color=get_color_scaled(0.75, False, True, False, False), label='0.75')
        green3 =  patches.Patch(color=get_color_scaled(1.0,  False, True, False, False), label='1.00')
        handles = handles + (purple1, purple2, purple3, green1, green2, green3)
        pyplot.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.10, 1.00))
        time.sleep(2)
        pyplot.savefig(saveFile, format='png')

    #--------------------------------------------------------------------------------
    # draw a slice through the current NN output in 2d
    #--------------------------------------------------------------------------------
    def Print2d(self, FixedIndex, Text):

        # get the 2d slice
        slice = self.OutData[10:30,10:30,FixedIndex,0]

        # make a title
        Title = Text + " Slice %d" % (FixedIndex)

        fig = pyplot.figure()
        pyplot.imshow(slice)
        pyplot.colorbar()
        pyplot.title(Title)
        pyplot.show()

    #--------------------------------------------------------------------------------
    # Sum across one access.  Display the result.
    # Summation axis:
    #     0 => x-axis
    #     1 => y-azis
    #     2 => z-axis
    #--------------------------------------------------------------------------------
    def Print2dProjection(self, SummationAxis, Text):

        # sum the NN output across x, y, or z axis
        Sum = np.zeros(shape=(NUM_PIXELS_1D, NUM_PIXELS_1D), dtype=float32)
        Sum = np.sum(self.OutData[:,:,:,0], axis=SummationAxis)

        # make a title
        axis = 'x'
        if SummationAxis == 0: axis = 'X'
        if SummationAxis == 1: axis = 'Y'
        if SummationAxis == 2: axis = 'Z'
        Title = Text + " Projection Across %s Axis" % (axis)

        # plot the sum
        fig = pyplot.figure()
        pyplot.imshow(Sum)
        pyplot.colorbar()
        pyplot.title(Title)
        pyplot.show()

    #--------------------------------------------------------------------------------
    # count the number of True pixels in voxels.
    #--------------------------------------------------------------------------------
    def CountNumberSet(self, voxels):
        Count = np.sum(voxels)
        return(Count)


    #--------------------------------------------------------------------------------
    # test of displaying voxels.
    # from: https://matplotlib.org/3.1.1/gallery/mplot3d/voxels.html
    #--------------------------------------------------------------------------------
    def Print3dTest(self):
        # prepare some coordinates
        x, y, z = np.indices((8, 8, 8))

        # draw cuboids in the top left and bottom right corners, and a link between them
        cube1 = (x < 3) & (y < 3) & (z < 3)
        cube2 = (x >= 5) & (y >= 5) & (z >= 5)
        link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

        # combine the objects into a single boolean array
        voxels = cube1 | cube2 | link

        # set the colors of each object
        colors = np.empty(voxels.shape, dtype=object)
        colors[link] = 'red'
        colors[cube1] = 'blue'
        colors[cube2] = 'green'

        # and plot everything
        fig = pyplot.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(voxels, facecolors=colors, edgecolor='k')

        pyplot.show()

    #--------------------------------------------------------------------------------
    # show the input or output map in 3d
    #--------------------------------------------------------------------------------
    def Print3d(self, InData, Channel=0):

        # C is grey, N is blue, O is red
        Color = 'xkcd:black'

        # draw InData[i,j,k,Channel] or OutData[i,j,k,Channel]
        fig = pyplot.figure()
        ax = Axes3D(fig)
        if (InData):
            (x, y, z) = np.nonzero(self.InData[:,:,:,Channel])
        else:
            (x, y, z) = np.nonzero(self.OutData[:,:,:,Channel])
        ax.scatter(x, y, z, c=Color, marker='s', s=60)
        pyplot.show()

    #--------------------------------------------------------------------------------
    # print a slice of density
    #
    # FixedAxis:
    #     0 => x-axis is fixed, print a Y-Z alice
    #     1 => y-azis is fixed, print a X-Z slice
    #     2 => z-axis is fixed, print a X-Y slice
    # FixedIndex:
    #     the fixed index in the slice
    # AtomType:
    #     C, N, O, H, or S  (C, N, or O for now)
    #     or X (show the connection channel)
    #     or A (show the 1st channel of a set of 4 channels)
    #--------------------------------------------------------------------------------
    def PrintSlice(self, FixedAxis, FixedIndex, InData, Channel=0):

        MaxVal = self.GetMaxVal(InData)

        InOutStr = "output"
        if InData: InOutStr = "input"

        if FixedAxis == 0:    print("Y-Z slice %d (%s)" % (FixedIndex, InOutStr))
        if FixedAxis == 1:    print("X-Z slice %d (%s)" % (FixedIndex, InOutStr))
        if FixedAxis == 2:    print("X-Y slice %d (%s)" % (FixedIndex, InOutStr))

        if FixedAxis == 0:
            i = FixedIndex
            for j in range(NUM_PIXELS_1D):
                Line = ""
                for k in range(NUM_PIXELS_1D):
                    Val = self.GetOneVal(i, j, k, InData, Channel)
                    Line = self.AddOneCharToLine(Val, Line, MaxVal)
                print(Line)

        if FixedAxis == 1:
            j = FixedIndex
            for i in range(NUM_PIXELS_1D):
                Line = ""
                for k in range(NUM_PIXELS_1D):
                    Val = self.GetOneVal(i, j, k, InData, Channel)
                    Line = self.AddOneCharToLine(Val, Line, MaxVal)
                print(Line)

        if FixedAxis == 2:
            k = FixedIndex
            for i in range(NUM_PIXELS_1D):
                Line = ""
                for j in range(NUM_PIXELS_1D):
                    Val = self.GetOneVal(i, j, k, InData, Channel)
                    Line = self.AddOneCharToLine(Val, Line, MaxVal)
                print(Line)

    #--------------------------------------------------------------------------------
    # helper function for PrintSlice
    # Val is in range -1:1.
    #--------------------------------------------------------------------------------
    def AddOneCharToLine(self, Val, Line, MaxVal):
        if MaxVal < 1.01:
            Val = int(Val * 100)
            if Val == 0:
                Line += "  . "
            elif Val < -99:
                Line += " -00"
            elif Val < 100:
                Line += " %3d" % (Val)
            else:
                Line = Line + "  00"
        elif MaxVal < 1001:
            if (Val > -.01) and (Val < 0.01):
                Line += "    ."
            elif Val < -999:
                Line += " -000"
            elif Val < 1000:
                Line += " %4d" % (Val)
            else:
                Line = Line + "  000"
        else:
            Line = Line + "???"
        return(Line)

    #--------------------------------------------------------------------------------
    # debugging: count the number of set pixels in each input and output channel
    #--------------------------------------------------------------------------------
    def CountSetPixels(self):

        #print("")
        for Channel in range(NUM_IN_CHANNELS):
            subset = self.InData[:,:,:,Channel]
            sum1 = (subset > SMALL_NUM).sum()
            sum2 = (subset > (1.0+SMALL_NUM)).sum()
            print("Input Channel %d: %d, %d" %(Channel, sum1, sum2))

        #print("")
        for Channel in range(NUM_OUT_CHANNELS):
            subset = self.OutData[:,:,:,Channel]
            sum1 = (subset > SMALL_NUM).sum()
            sum2 = (subset > (1.0+SMALL_NUM)).sum()
            print("Output Channel %d: %d, %d" %(Channel, sum1, sum2))

        # the numpy functions above are WAY faster than the code I wrote below!
        return

    #--------------------------------------------------------------------------------
    # remake the C, N, and O electron density maps for 1 time step.
    # use the saved atom coordinates.
    # this function allows us to change atom-radius mid-stream.
    #--------------------------------------------------------------------------------
    def ReMakeMap(self, InData, AtomRadiusSqr, Channel=0):

        # get the saved positions
        (AtomTypes, AtomPositions) = self.GetRawData()

        # for keeping track of atom overlap
        NumPixelsTouchedInOutputDensity = 0
        NumDuplicatesInOutputDensity = 0

        # clear input or output map
        if InData:
            self.InData.fill(0.0)
        else:
            self.OutData.fill(0.0)

        #--------------------------------------------------------------
        # for each atom, add pixel-occupancies to density maps.
        #--------------------------------------------------------------
        for i in range(len(AtomTypes)):
            # add density for an atom to InData or OutData.
            (NumPixels, NumDuplicates) = self.AddToDensity(AtomPositions[i], InData, AtomRadiusSqr, Channel)
            NumPixelsTouchedInOutputDensity += NumPixels
            NumDuplicatesInOutputDensity += NumDuplicates

        return((NumPixelsTouchedInOutputDensity, NumDuplicatesInOutputDensity))

    #--------------------------------------------------------------------------------
    # add random noise in range [-MapNoise : MapNoise] to map pixels.
    # clip map at [0 : 1].
    #--------------------------------------------------------------------------------
    def AddMapNoise(self, InData, MapNoise, Channel=0):

        if InData:
            Noise = np.random.rand(NUM_PIXELS_1D, NUM_PIXELS_1D, NUM_PIXELS_1D)
            Noise = Noise * 2 * MapNoise - MapNoise
            self.InData[:,:,:,Channel] += Noise
        else:
            Noise = np.random.rand(NUM_PIXELS_1D, NUM_PIXELS_1D, NUM_PIXELS_1D)
            Noise = Noise * 2 * MapNoise - MapNoise
            self.OutData[:,:,:,Channel] += Noise

        test = 1

    #--------------------------------------------------------------------------------
    # calculate the FFT of OutData.
    # calculate magnitudes at each point.
    # put this on InData.
    #--------------------------------------------------------------------------------
    def Calc_FFT_Mags(self):

        # calculate the FFT of OutData channel 0
        FFT = np.fft.fftn(self.OutData[:,:,:,0])

        # calculate magnitudes from real and imaginary parts
        Mags = np.sqrt(np.square(np.real(FFT)) + np.square(np.imag(FFT)))

        # check that IFFT(FFT(OutData)) = OutData.
        # it looks right.
        # IFFT = np.real(np.fft.ifftn(FFT))
        # self.InData[:,:,:,0] = IFFT
        # self.PrintSlice(2, 50, True)

        # put the real and imaginary parts of FFT on output channels 0 and 1
        self.InData[:,:,:,0] = np.real(FFT)
        self.InData[:,:,:,1] = np.imag(FFT)

    #--------------------------------------------------------------------------------
    # make a Patterson Map from OutData and put it on InData
    # the Patterson Map is:
    #    IFFT(FFT(density) * complex-conjugate(FFT(density)))
    # see my 6/2019 - ?  notebook, p. 3.
    #--------------------------------------------------------------------------------
    def MakePattersonMap(self, outputIsWorkArray=False):

        # calculate the FFT of OutData channel 0
        FFT = np.fft.fftn(self.OutData[:,:,:,0])

        # calculate the complex-conjugate of the FFT
        FFT_CONJ = np.conj(FFT)

        # the autocorrelation is the IFFT(FFT * FFT_CONJ)
        AUTO_CORR = np.fft.ifftn(FFT * FFT_CONJ)
        # AUTO_CORR = np.fft.ifftn(FFT)   # use this instead of above to use output data on the input

        # AUTO_CORR should only have real components. in any case, just use the real part.
        TEMP = np.real(AUTO_CORR)

        # get rid of self-hit peaks
        # t1 = (TEMP > 0).sum()
        # TEMP[TEMP > 30] = 0     # this is good for PIXEL_LEN = 0.32
        # TEMP[TEMP > 100] = 0    # this is good for PIXEL_LEN = 0.64
        # TEMP[TEMP > 6] = 0      # this is good for PIXEL_LEN = 0.64 and OnlyCAlphas
        # t2 = (TEMP > 0).sum()

        # swap quadrants so self-correlation is centered
        TEMP = np.fft.fftshift(TEMP)

        # this scales the NN input. comment out to use raw data on the input
        # max = TEMP.max() / 50
        # TEMP = np.tanh(TEMP/max)

        # put this on InData channel 0
        if outputIsWorkArray:
            self.WorkArray[:,:,:,0] = TEMP
        else:
            self.InData[:,:,:,0] = TEMP

    #--------------------------------------------------------------------------------
    # calculate a similarity score between:
    #   1. the true Patterson Map on the NN input
    #   2. the test Patterson Map on the WorkArray
    #--------------------------------------------------------------------------------
    def ComparePattersonMaps(self):
        MSE = ((self.InData - self.WorkArray)**2).mean()
        return(MSE)

    #--------------------------------------------------------------------------------
    # return the average position of Atoms
    #--------------------------------------------------------------------------------
    def CalcCenter(self, atomPositions):
        Center = np.zeros(shape=[3], dtype=float32)
        for i in range(len(atomPositions)):
            Center += atomPositions[i]
        Center = Center / float(len(atomPositions))
        return(Center)

    #--------------------------------------------------------------------------------
    # return the centro-symmetric inversion of atomPositions
    #--------------------------------------------------------------------------------
    def GetCentroSym(self, atomPositions, atomTypes):
        newAtomPositions = []
        newAtomTypes = []
        for i in range(len(atomPositions)):
            newAtomPositions.append(atomPositions[i] * -1.0)
            newAtomTypes.append(atomTypes[i])
        return((newAtomPositions, newAtomTypes))

    #--------------------------------------------------------------------------------
    # add the centro-symmetric inversion of atomPositions to atomPositions
    #--------------------------------------------------------------------------------
    def AddCentroSym(self, atomPositions, atomTypes):
        newAtomPositions = []
        newAtomTypes = []
        for i in range(len(atomPositions)):
            newAtomPositions.append(atomPositions[i])
            newAtomPositions.append(atomPositions[i] * -1.0)
            newAtomTypes.append(atomTypes[i])
            newAtomTypes.append(atomTypes[i])
        return((newAtomPositions, newAtomTypes))

    #--------------------------------------------------------------------------------
    # return a new set of atomPositions with the probability that each
    # coordinate is inverted is 50%
    #--------------------------------------------------------------------------------
    def RandomizePositions(self, atomPositions):

        newSet = []
        for position in atomPositions:
            val = np.random.uniform(0, 1)
            if val < 0.5:
                newSet.append(position*-1.0)
            else:
                newSet.append(position)
        return(newSet)

    #--------------------------------------------------------------------------------
    # add the centro-symmetric image of atomPositions to atomPositions.
    #--------------------------------------------------------------------------------
    def GetDoubleRandomPositions(self, NumAtoms, AtomRadiusSqr):
        AtomRadius = np.sqrt(AtomRadiusSqr)

        # repeat until the atoms are adequately spaced
        TooClose = True
        while (TooClose):
            # the new atom positions are the old ones plus their negatives
            atomPositions = self.GetRandomPositions(NumAtoms, AtomRadiusSqr)
            newAtomPositions = []
            for i in range(len(atomPositions)):
                newAtomPositions.append(atomPositions[i])
                newAtomPositions.append(atomPositions[i] * -1.0)

            if not self.PointsTooClose(newAtomPositions, AtomRadius):
                TooClose = False
        return(newAtomPositions)

    #--------------------------------------------------------------------------------
    # This is a check on GetRandomPositions2.
    # This routine rejects a set of atoms if any are outside a bounding box.
    # Need this because the set of atoms returned in GetRandomPositions2 is
    # centered after the bounding-box checks in that routine.
    #--------------------------------------------------------------------------------
    def GetRandomPositions3(self, NumAtoms, AtomRadiusSqr):
        BARRIER_MIN = -18.0
        BARRIER_MAX =  18.0

        # repeat until we get a good set
        GotOne = False
        while not GotOne:
            # get a set of atom positions
            atomPositions = self.GetRandomPositions2(NumAtoms, AtomRadiusSqr)
            # make array indicating if any x, y, or z is outside the BARRIER box
            checkArray = np.logical_or((atomPositions<BARRIER_MIN), (atomPositions>BARRIER_MAX))
            # if any coordinate is outside the BARRIER box, reject the set
            if np.any(checkArray):
                continue
            # we got a good set
            GotOne = True
        return(atomPositions)

    #--------------------------------------------------------------------------------
    # new and improved GetRandomPositions
    # get NumAtoms random coordinates inside the box
    # make sure they're at least 3*AtomRadius apart and inside a smaller box
    #--------------------------------------------------------------------------------
    def GetRandomPositions2(self, NumAtoms, AtomRadiusSqr):
        # MIN = -5.7
        # MAX =  5.7
        MIN = -7.125     # these are the values used for model/training 1/86-90 and 8/1-21
        MAX =  7.125     # .
        # MIN = -7.2
        # MAX =  7.2
        # MIN = -9.0
        # MAX =  9.0
        # MIN = -13.0    # I'm testing if I really need the small box
        # MAX =  13.0    # .
        AtomRadius = np.sqrt(AtomRadiusSqr)

        atomPositions = []

        # repeat until there are NumAtoms in atomPositions
        while len(atomPositions) < NumAtoms:

            # add a candidate random coordinate in the box to atomPositions
            x = np.random.uniform(MIN + AtomRadius, MAX - AtomRadius)
            y = np.random.uniform(MIN + AtomRadius, MAX - AtomRadius)
            z = np.random.uniform(MIN + AtomRadius, MAX - AtomRadius)
            Pt = np.array([float32(x), float32(y), float32(z)], dtype=float32)
            atomPositions.append(Pt)

            # if this point is too close to a point already in the box
            if self.IsLastPointTooClose(atomPositions, AtomRadius):
                # remove the last point from atomPositions
                del atomPositions[len(atomPositions)-1]

        # translate the atoms so they're centered at (0,0,0)
        Center = self.CalcCenter(atomPositions)
        atomPositions = atomPositions - Center
        Center = self.CalcCenter(atomPositions)
        return(atomPositions)

    #--------------------------------------------------------------------------------
    # get NumAtoms random coordinates inside the box
    # make sure they're at least 3*AtomRadius apart and inside a smaller box
    #--------------------------------------------------------------------------------
    def GetRandomPositions(self, NumAtoms, AtomRadiusSqr):
        MIN = -5.7
        MAX =  5.7
        AtomRadius = np.sqrt(AtomRadiusSqr)

        # repeat until the atoms are adequately spaced
        TooClose = True
        while (TooClose):
            # get NumAtoms random coordinates
            atomPositions = []
            while len(atomPositions) < NumAtoms:
                # get a candidate random coordinate in the box
                x = np.random.uniform(MIN + AtomRadius, MAX - AtomRadius)
                y = np.random.uniform(MIN + AtomRadius, MAX - AtomRadius)
                z = np.random.uniform(MIN + AtomRadius, MAX - AtomRadius)
                Pt = np.array([float32(x), float32(y), float32(z)], dtype=float32)
                atomPositions.append(Pt)
            if not self.PointsTooClose(atomPositions, AtomRadius):
                TooClose = False

        # translate the atoms so they're centered at (0,0,0)
        Center = self.CalcCenter(atomPositions)
        atomPositions = atomPositions - Center
        Center = self.CalcCenter(atomPositions)
        return(atomPositions)

    #----------------------------------------------------------------------------------------------------------------
    # return True if the last atomPosition is less than ATOM_RADIUS_MULTIPLIER*AtomRadius from another atomPosition
    #----------------------------------------------------------------------------------------------------------------
    def IsLastPointTooClose(self, atomPositions, AtomRadius):

        ATOM_RADIUS_MULTIPLIER = 3.0

        j = len(atomPositions) - 1
        for i in range(len(atomPositions)-1):
            if np.linalg.norm(atomPositions[i] - atomPositions[j]) < (AtomRadius * ATOM_RADIUS_MULTIPLIER):
                return(True)
        return(False)

    #---------------------------------------------------------------------------------------------------
    # return True if 2 points in atomPositions are less than ATOM_RADIUS_MULTIPLIER*AtomRadius apart
    #---------------------------------------------------------------------------------------------------
    def PointsTooClose(self, atomPositions, AtomRadius):

        ATOM_RADIUS_MULTIPLIER = 3.0

        for i in range(len(atomPositions)-1):
            for j in range(i+1, len(atomPositions)):
                if np.linalg.norm(atomPositions[i] - atomPositions[j]) < (AtomRadius * ATOM_RADIUS_MULTIPLIER):
                    return(True)
        return(False)

    #--------------------------------------------------------------------------------
    # make the electron density maps for either input or output.
    #--------------------------------------------------------------------------------
    def MakeMap(self, AtomPositions, AtomTypes, InData, AtomRadiusSqr, Channel=0, SavePositions=True, ClearData=False):

        assert(len(AtomPositions) == len(AtomTypes))

        if ClearData:
            if InData:
                self.InData.fill(0.0)
            else:
                self.OutData.fill(0.0)

        # save these so we can refer back to it later
        if SavePositions:
            self.AtomPositions = AtomPositions
            self.AtomTypes = AtomTypes

        NumPixelsTouchedInOutputDensity = 0
        NumDuplicatesInOutputDensity = 0

        #--------------------------------------------------------------
        # for each atom, add pixel-occupancies to density maps
        #--------------------------------------------------------------
        for i in range(len(AtomTypes)):
            assert(AtomTypes[i]=='C' or AtomTypes[i]=='N' or AtomTypes[i]=='O')
            # add density for an atom to InData or OutData.
            (NumPixels, NumDuplicates) = self.AddToDensity(AtomPositions[i], InData, AtomRadiusSqr, Channel)
            NumPixelsTouchedInOutputDensity += NumPixels
            NumDuplicatesInOutputDensity += NumDuplicates

        return((NumPixelsTouchedInOutputDensity, NumDuplicatesInOutputDensity))

    #--------------------------------------------------------------------------------
    # calculate the occupancy for the pixel at AtomPos, and its 26 neighbors.
    # add this to an electron density map
    #--------------------------------------------------------------------------------
    def AddToDensity(self, AtomPos, InData, AtomRadiusSqr, Channel=0, SubtractInstead=False):

        # get the pixel that contains the atom-center of AtomPos
        (i,j,k) = self.GetPixel(AtomPos)

        # in case we want to look at the number of pixels used to determine pixel-occupancy
        NumCubes = [0]

        # a check. totalFraction should be ((4/3)*pi*(radius**3)) / (27*(pixel-length**3))
        totalOccupiedVol = 0

        NumPixelsWithOccupancy = 0
        NumDuplicates = 0

        BoxRange = 2
        if AtomRadiusSqr < (PIXEL_LEN*PIXEL_LEN + SMALL_NUM):
            BoxRange = 1

        # for the 3x3x3 block of pixels centered at (i,j,k)
        for di in range(-BoxRange,BoxRange+1):
            for dj in range(-BoxRange,BoxRange+1):
                for dk in range(-BoxRange,BoxRange+1):
                    (ii, jj, kk) = (i+di, j+dj, k+dk)
                    # if the pixel has some occupancy
                    #  if self.DoesPixelHaveSomeOccupancy(AtomPos, (ii, jj, kk)):
                    # determine the fraction of the pixel occupied by the atom sphere
                    PixelPos000 = self.Get000Pos((ii, jj, kk))

                    # tried several different methods
                    # occupiedVolPerPixelVol = CalcPixelOccupancy(AtomPos, PixelPos000, 4, NumCubes, AtomRadiusSqr)
                    # occupiedVolPerPixelVol = CalcPixelOccupancyNoRecursion(AtomPos, PixelPos000, NumCubes, AtomRadiusSqr)
                    # occupiedVolPerPixelVol = CalcPixelOccupancyUsingRandomNumbers(AtomPos, PixelPos000, 8000, AtomRadiusSqr)
                    occupiedVolPerPixelVol = CalcPixelOccupancyUsingPtsOnGrid(AtomPos, PixelPos000, AtomRadiusSqr)

                    occupiedVol = occupiedVolPerPixelVol * PIXEL_LEN_CUBED
                    totalOccupiedVol = totalOccupiedVol + occupiedVol

                    # add it to the proper density map
                    GetVal = self.GetOneVal(ii, jj, kk, InData, Channel)
                    if SubtractInstead:
                        SetVal = GetVal - occupiedVolPerPixelVol
                    else:
                        SetVal = GetVal + occupiedVolPerPixelVol
                    self.SetOneVal(ii, jj, kk, InData, SetVal, Channel)

                    if occupiedVolPerPixelVol > SMALL_NUM:
                        NumPixelsWithOccupancy = NumPixelsWithOccupancy + 1
                        if GetVal > SMALL_NUM:
                            NumDuplicates = NumDuplicates + 1

        # this is for debugging. it's ok if it's slow since it will be turned off at some point.
        AtomRadiusCubed = AtomRadiusSqr * np.sqrt(AtomRadiusSqr)
        totalFraction = totalOccupiedVol / (27 * PIXEL_LEN_CUBED)
        exactFraction = 4/3 * PI * AtomRadiusCubed / (27 * PIXEL_LEN_CUBED)

        return((NumPixelsWithOccupancy, NumDuplicates))

    #--------------------------------------------------------------------------------
    # see my 2018 notebook, p. 102.
    # AtomPos is an (x,y,z) tuple
    # TestPixel is an (i,j,k) tuple
    # Determine if TestPixel has any occupancy due to the atom at AtomPos.
    #--------------------------------------------------------------------------------
    def DoesPixelHaveSomeOccupancy(self, AtomPos, TestPixel, AtomRadiusSqr):

        # get the pixel that contains the atom-center
        AtomPixel = self.GetPixel(AtomPos)

        # make sure TestPixel is a neighbor of AtomPixel
        idiff = TestPixel[0] - AtomPixel[0]
        jdiff = TestPixel[1] - AtomPixel[1]
        kdiff = TestPixel[2] - AtomPixel[2]
        abs_idiff = abs(idiff)
        abs_jdiff = abs(jdiff)
        abs_kdiff = abs(kdiff)
        assert(abs_idiff <= 1 and abs_jdiff <= 1 and abs_kdiff <= 1)
        Sum = abs_idiff + abs_jdiff + abs_kdiff

        # Sum = 1 => (1,0,0) neighbor  (6 of these)
        # Sum = 2 => (1,1,0) neighbor (12 of these)
        # Sum = 3 => (1,1,1) neighbor  (8 of these)

        # if it's the atom-center pixel, return True.
        if (Sum == 0):
            return(True)

        # if it's a (1,0,0) neighbor, return True. See my 2018 notebook, p. 102.
        if (Sum == 1):
            return(True)

        # if it's a (1,1,0) neighbor
        if (Sum == 2):
            # get the vertices of the shared edge between AtomPixel and TestPixel
            i1 = i2 = AtomPixel[0]
            j1 = j2 = AtomPixel[1]
            k1 = k2 = AtomPixel[2]
            if idiff == 0:       i2 = AtomPixel[0] + 1
            if idiff == 1:  i1 = i2 = AtomPixel[0] + 1
            if jdiff == 0:       j2 = AtomPixel[1] + 1
            if jdiff == 1:  j1 = j2 = AtomPixel[1] + 1
            if kdiff == 0:       k2 = AtomPixel[2] + 1
            if kdiff == 1:  k1 = k2 = AtomPixel[2] + 1

            # get 2 points that define the shared edge
            Pt1 = self.Get000Pos((i1,j1,k1))
            Pt2 = self.Get000Pos((i2,j2,k2))

            # get the distance between line Pt1->Pt2 and AtomPos
            P1 = np.array(Pt1)
            P2 = np.array(Pt2)
            P3 = np.array(AtomPos)
            vec = np.cross(P2-P1, P3-P1)
            #dist = np.linalg.norm(vec) / np.linalg.norm(P2-P1)
            v0 = vec[0]
            v1 = vec[1]
            v2 = vec[2]
            topSqr = v0*v0 + v1*v1 + v2*v2
            d0 = P2[0] - P1[0]
            d1 = P2[1] - P1[1]
            d2 = P2[2] - P1[2]
            botSqr = d0*d0 + d1*d1 + d2*d2
            distSqr = topSqr / botSqr

            # if dist < AtomRadius, then some of the pixel is within the sphere
            if (distSqr < AtomRadiusSqr):
                return(True)
            return(False)

        # if it's a (1,1,1) neighbor
        if (Sum == 3):
            # get the one shared vertex between AtomPixel and TestPixel
            i1 = AtomPixel[0]
            j1 = AtomPixel[1]
            k1 = AtomPixel[2]
            if (idiff == 1): i1 = i1+1
            if (jdiff == 1): j1 = j1+1
            if (kdiff == 1): k1 = k1+1

            # get the position of the shared vertex
            Pt = self.Get000Pos((i1,j1,k1))

            # get the distance between Pt and AtomPos
            #P1 = np.array(Pt)
            #P2 = np.array(AtomPos)
            #dist = np.linalg.norm(P1-P2)
            (d0,d1,d2) = np.subtract(Pt, AtomPos)
            distSqr = d0*d0 + d1*d1 + d2*d2

            # if dist < AtomRadius, then some of the pixel is within the sphere
            #if (dist < AtomRadius):
            if (distSqr < AtomRadiusSqr):
                return(True)
            return(False)

    #--------------------------------------------------------------------------------
    # get the (x,y,z) position of the center of Pixel
    #--------------------------------------------------------------------------------
    def GetCenterPos(self, Pixel):
        (i,j,k) = Pixel
        x = BOX_MIN + i*PIXEL_LEN + PIXEL_LEN/2.0
        y = BOX_MIN + j*PIXEL_LEN + PIXEL_LEN/2.0
        z = BOX_MIN + k*PIXEL_LEN + PIXEL_LEN/2.0
        return((x,y,z))

    #--------------------------------------------------------------------------------
    # get the (x,y,z) position of the (0,0,0) corner of Pixel
    # this is the inverse of GetPixel(), but more specific.
    #--------------------------------------------------------------------------------
    def Get000Pos(self, Pixel):
        (i, j, k) = Pixel
        x = BOX_MIN + i*PIXEL_LEN
        y = BOX_MIN + j*PIXEL_LEN
        z = BOX_MIN + k*PIXEL_LEN
        return((x,y,z))

    #--------------------------------------------------------------------------------
    # (x,y,z) -> (i,j,k)
    #
    # pass: Pos, an (x,y,z) tuple
    # return: an (i,j,k) tuple
    #
    # x, y, z are in the range [BOX_MIN : BOX_MAX]
    # i, j, k are in the range [0 : NUM_PIXELS_1D - 1]
    #
    # note that the same (i,j,k) is returned for:
    #     x_start <= x < (x_start + PIXEL_SIZE)
    #     y_start <= y < (y_start + PIXEL_SIZE)
    #     z_start <= z < (z_start + PIXEL_SIZE)
    # where (x_start, y_start, z_start) is the (0,0,0) corner of the pixel
    #--------------------------------------------------------------------------------
    def GetPixel(self, Pos):

        (x,y,z) = Pos

        # make sure (x,y,z) are in range
        assert(x > BOX_MIN and y > BOX_MIN and z > BOX_MIN)
        assert(x < BOX_MAX and y < BOX_MAX and z < BOX_MAX)

        # translate so x, y, z are in the range [0 : BOX_MAX - BOX_MIN]
        x = x - BOX_MIN
        y = y - BOX_MIN
        z = z - BOX_MIN

        # convert to (i,j,k)
        i = int(x / PIXEL_LEN)
        j = int(y / PIXEL_LEN)
        k = int(z / PIXEL_LEN)

        # make sure (i,j,k) are in range
        #np_1d = NUM_PIXELS_1D
        #assert(i >= 0    and j >= 0    and k >= 0)
        #assert(i < np_1d and j < np_1d and k < np_1d)

        return((i,j,k))

    #--------------------------------------------------------------------------------
    #  set the NN input and output data for one training example.
    #--------------------------------------------------------------------------------
    def SetOneTrainingData(self, Lines, molNum, simNum, stepNum, AtomRadiusSqr,
                           doRotation=False, printNumPixelsTouchedInOutputDensity=False, DisplacementNoise=0.0,
                           OnlyCAlphas=False, NthCAlpha=1, NumAtoms=-1, NumResidues=-1, AddSymmetryAtoms=False,
                           batchType='training'):

        # clear the current NN data
        self.ClearNN()

        # check that the time steps is valid.
        if not Lines.IsValid(molNum, simNum, stepNum):
            print("error: requested data for an invalid time step")
            return

        # get the raw-data center
        Center = Lines.GetProteinCenter(molNum, simNum, stepNum)

        # get resTypes for the training data
        (ResTypes, ResNumbers) = Lines.GetResTypesAndResNumbers(molNum, simNum, stepNum, OnlyCAlphas, NthCAlpha)

        # get atomTypes for the training data
        atomTypes = Lines.GetAtomTypes(molNum, simNum, stepNum, OnlyCAlphas, NthCAlpha)

        # get atom positions for the training data
        # atom postions are lists of numpy (x,y,z) arrays.
        # translate each atom s.t. the time-step is centered.
        atomPositions = Lines.GetProteinCoordinatesTranslated(molNum, simNum, stepNum, Center, OnlyCAlphas, NthCAlpha)

        # if NumResidues is specified, use a NumResidues subset of the complete set of residues
        if NumResidues > 0:
            TotalNumResidues = ResNumbers[len(ResNumbers)-1]                      # 20

            # get the first and last residue of the peptide subset
            FirstResNum = np.random.randint(1, TotalNumResidues-NumResidues+2)    #  1:18
            LastResNum = FirstResNum + NumResidues - 1                            #  3:20

            # make a mask that is true for just these few residues
            Maskable = np.asarray(ResNumbers)
            Mask = (Maskable >= FirstResNum) & (Maskable <= LastResNum)

            # get the center CAlpha position
            index_into_CAlphaIndices = int((FirstResNum+LastResNum)/2 - 1)
            CenterCAlphaPos = atomPositions[CAlphaIndices[index_into_CAlphaIndices]]

            # make subsets with only the atoms in the Mask
            newAtomTypes = []
            newAtomPositions = []
            for i in range(len(atomTypes)):
                if Mask[i]:
                    newAtomTypes.append(atomTypes[i])
                    newAtomPositions.append(atomPositions[i])
            atomTypes = newAtomTypes
            atomPositions = newAtomPositions

            # translate atomPositions s.t. they're centered on the central CAlpha
            atomPositions = atomPositions - CenterCAlphaPos

            # optionally rotate the atom positions
            if (doRotation):
                (Angle1, Angle2, Angle3) = Get3RandomAngles()
                atomPositions = RotateProtein(atomPositions, Angle1, Angle2, Angle3)

            # translate the atoms to the center of the (+,+,+) quadrant of the big box
            # QuadrantCenter = np.array([BOX_MAX/2, BOX_MAX/2, BOX_MAX/2], dtype=float32)
            # atomPositions = atomPositions + QuadrantCenter

            # make the NN output density map for atomPositions
            (NumPixelsTouchedInOutputDensity, NumDuplicatesInOutputDensity) = \
                self.MakeMap(atomPositions, atomTypes, False, AtomRadiusSqr, Channel=0, SavePositions=True)

            # make the Patterson map from the output density map and put it on the NN input
            self.MakePattersonMap()

            # get the centro-symmetric inversion of atomPositions
            if AddSymmetryAtoms:
                (atomPositions, atomTypes) = self.GetCentroSym(atomPositions, atomTypes)

            # add the centro-symmetric atoms to the output density map
            self.MakeMap(atomPositions, atomTypes, False, AtomRadiusSqr, Channel=0, SavePositions=False)

            # divide the output density by 2 because it can exceed 1.0
            self.OutData[:,:,:,0] /= 2.0

            return((atomTypes, atomPositions))

        # if NumAtoms is specified, use random atom positions instead of atom positions from the MD simulations
        if NumAtoms > 0:

            # get NumAtoms random atoms in the box defined in "GetRandomPositions"
            # atomPositions = self.GetRandomPositions(NumAtoms, AtomRadiusSqr)
            atomPositions = self.GetRandomPositions3(NumAtoms, AtomRadiusSqr)

            # the atom-types don't matter for this
            atomTypes = atomTypes[0] * NumAtoms

            # make the NN output density map for atomPositions
            (NumPixelsTouchedInOutputDensity, NumDuplicatesInOutputDensity) = \
                self.MakeMap(atomPositions, atomTypes, False, AtomRadiusSqr, Channel=0, SavePositions=True)

            # make the Patterson map from the output density map and put it on the NN input
            self.MakePattersonMap()

            # get the centro-symmetric inversion of atomPositions
            # ---------- comment out these lines to skip adding centro-symmetric atoms. ----------#
            if AddSymmetryAtoms:                                                                  #
                (atomPositions, atomTypes) = self.GetCentroSym(atomPositions, atomTypes)          #
            # ---------- comment out these lines to skip adding centro-symmetric atoms. ----------#

            # add the centro-symmetric atoms to the output density map
            # --------------- comment out this line to skip adding centro-symmetric atoms. ----------------#
            self.MakeMap(atomPositions, atomTypes, False, AtomRadiusSqr, Channel=0, SavePositions=False)   #
            # --------------- comment out this line to skip adding centro-symmetric atoms. ----------------#

            # divide the output density by 2 because it can exceed 1.0
            self.OutData[:,:,:,0] /= 2.0

            return((atomTypes, atomPositions))

        # optionally rotate the atom positions
        if (doRotation):
            (Angle1, Angle2, Angle3) = Get3RandomAngles()
            atomPositions = RotateProtein(atomPositions, Angle1, Angle2, Angle3)

        # add random noise to the NN input coordinates
        if DisplacementNoise > SMALL_NUM:
            atomPositions = AddDisplacementNoise(atomPositions, DisplacementNoise)

        # make the output density map (on channel 0 of the output)
        (NumPixelsTouchedInOutputDensity, NumDuplicatesInOutputDensity) = \
            self.MakeMap(atomPositions, atomTypes, False, AtomRadiusSqr, Channel=0, SavePositions=True)

        # calculate a Patterson Map of OutData and put in on InData
        self.MakePattersonMap()

        if printNumPixelsTouchedInOutputDensity:
            print("num pixel additions making output density map = %d (num shared pixels = %d)" \
                  % (NumPixelsTouchedInOutputDensity, NumDuplicatesInOutputDensity))

        return((atomTypes, atomPositions))

    #--------------------------------------------------------------------------------
    # print histogram of Input or Output.
    #--------------------------------------------------------------------------------
    def PrintHistogram(self, InData, numBins, whatIsIt, Channel=0):

        if (whatIsIt != ''):
            print("%s histogram:" % (whatIsIt))

        if InData:
            Lo = self.InData[:,:,:,Channel].min()
            Hi = self.InData[:,:,:,Channel].max()
        else:
            Lo = self.OutData[:,:,:,Channel].min()
            Hi = self.OutData[:,:,:,Channel].max()

        Counts = [0] * numBins
        Delta = (Hi - Lo) / float(numBins)

        # get the channel
        if InData:
            subset = self.InData[:,:,:,Channel]
        else:
            subset = self.OutData[:,:,:,Channel]

        print("min = %f, max = %f" % (subset.min(), subset.max()))

        # subset = subset / subset.max()

        # the 1st bin
        LowVal = Lo
        HiVal = Lo + Delta

        # for each bin
        Total = 0
        for i in range(numBins):
            # count the number of pixels in the bin range
            Count1 = (subset < LowVal).sum()
            Count2 = (subset <= HiVal).sum()
            Counts[i] = Count2 - Count1
            LowVal = HiVal
            HiVal += Delta
            Total += Counts[i]

        if InData:
            print("Histogram for InData:", end="")
        else:
            print("Histogram for OutData:", end="")
        print(Counts, end="")

        Count1 = (subset <  SMALL_NUM).sum()
        nx = subset.shape[0]
        ny = subset.shape[1]
        nz = subset.shape[2]
        print(" num-zero-or-less = %d, num-pos = %d" % (Count1, nx*ny*nz - Count1))

    #--------------------------------------------------------------------------------
    # calculate the sum of the (i,j,k) pixel and all its neighbors
    #--------------------------------------------------------------------------------
    def CalcDensitySum(self, InData, i, j, k, Channel=0):
        Sum = 0.0
        for ii in range(-1,2):
            for jj in range(-1,2):
                for kk in range(-1,2):
                    Sum += self.GetOneVal(i+ii, j+jj, k+kk, InData, Channel)
        return(Sum)

    #--------------------------------------------------------------------------------
    # return the smallest atom-pair distance
    #--------------------------------------------------------------------------------
    def GetClosestPairDist(self):
        closest = 999
        for i in range(len(self.AtomPositions)-1):
            for j in range(i+1, len(self.AtomPositions)):
               dist = np.linalg.norm(self.AtomPositions[i] - self.AtomPositions[j])
               if dist < closest: closest = dist
        return(closest)

    #--------------------------------------------------------------------------------
    # return the largest atom-pair distance
    #--------------------------------------------------------------------------------
    def GetFarthestPairDist(self):
        farthest = -999
        for i in range(len(self.AtomPositions)-1):
            for j in range(i+1, len(self.AtomPositions)):
               dist = np.linalg.norm(self.AtomPositions[i] - self.AtomPositions[j])
               if dist > farthest: farthest = dist
        return(farthest)

    #--------------------------------------------------------------------------------
    # return the distance of the atom that's farthest from the center
    #--------------------------------------------------------------------------------
    def GetFarthestFromCenter(self):
        farthest = -999
        for i in range(len(self.AtomPositions)-1):
            dist = np.linalg.norm(self.AtomPositions[i])
            if dist > farthest: farthest = dist
        return(farthest)

    #--------------------------------------------------------------------------------
    # for each pixel, estimate its position.
    # use a weighted average of the pixel and its surrounding pixels.
    #--------------------------------------------------------------------------------
    def EstimateAtomPositions(self, Pixels):

        Positions = []

        # for each pixel
        for pixel in Pixels:
            # for the pixel and its neighbors
            AvgPos = 0
            TotalWeight = 0
            (i, j, k) = pixel
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    for kk in range(-1, 2):
                        # make a weighted average of the pixel position
                        Pos = np.array(self.GetCenterPos((i+ii, j+jj, k+kk)))
                        Weight = self.GetOneVal(i+ii, j+jj, k+kk, False)
                        AvgPos += Pos * Weight
                        TotalWeight += Weight
            # save the weighted average position
            AvgPos /= TotalWeight
            Positions.append(AvgPos)

        return(np.array(Positions))

    #--------------------------------------------------------------------------------
    # look through each pixel in the output image for:
    #   1. pixel values over Threshold1 that are also a local maximum
    #   2. pixel-plus-neighbor-sums over Threshold2
    # return: the pixels
    # return: the pixel values
    # return: the sum of the pixel and its neighbors
    # note: if no Threshold2 is given, then only condition 1 needs to be met.
    #--------------------------------------------------------------------------------
    def GetPeaks1(self, Threshold1, Threshold2=-1.0):

        Pixels = []
        PixelVals = []
        PixelSums = []

        # look through all pixels in the output image
        for i in range(NUM_PIXELS_1D):
            for j in range(NUM_PIXELS_1D):
                for k in range(NUM_PIXELS_1D):
                    # if pixel > Threshold
                    pixelDensity = self.GetOneVal(i, j, k, False)
                    if pixelDensity > Threshold1:
                        # if pixel is > all its neighbors and Sum > Threshold2
                        (Answer, Sum) = self.CenterPixelIsGreatestDensity(False, i, j, k)
                        if Answer == True and Sum > Threshold2:
                            # save this pixel
                            pixel = (i, j, k)
                            Pixels.append(pixel)
                            PixelVals.append(pixelDensity)
                            PixelSums.append(Sum)
        return(Pixels, PixelVals, PixelSums)

    #--------------------------------------------------------------------------------
    # look through each pixel in the output image for:
    #   1. pixel-values over threshold1
    #   2. pixel-plus-neighbor-sums over Threshold2
    # return: the pixels
    # return: the pixel values
    # return: the sum of the pixel and its neighbors
    # NOT USED. This function relaxes the local-maximum condition of GetPeaks1
    #--------------------------------------------------------------------------------
    def GetPeaks2(self, Threshold1, Threshold2):

        Pixels = []
        PixelVals = []
        PixelSums = []

        # look through all pixels in the output image
        for i in range(NUM_PIXELS_1D):
            for j in range(NUM_PIXELS_1D):
                for k in range(NUM_PIXELS_1D):
                    # if pixel > Threshold1
                    pixelDensity = self.GetOneVal(i, j, k, False)
                    if pixelDensity > Threshold1:
                        # if neighbor-sum > Threshold2:
                        (Answer, Sum) = self.CenterPixelIsGreatestDensity(False, i, j, k)
                        if Sum > Threshold2:
                            pixel = (i,j,k)
                            Pixels.append(pixel)
                            PixelVals.append(pixelDensity)
                            PixelSums.append(Sum)
        return(Pixels, PixelVals, PixelSums)

    #-----------------------------------------------------------------------------------------------
    # for each pixel in KnownPixels:
    #   look at all pixel's nearest neighbors
    #   if  1) neighbor-pixel > Threshold1 AND
    #       2) neighbor-pixel > all its neighbors except pixel AND
    #       3) sum of neigbor-pixel and its neighbors > Threshold2
    #   then: return info for this neighbor
    # return: the neighbor pixel
    # return: the neighbor pixel-value
    # return: the sum of the neighbor-pixel and its neighbors (in all directions)
    # NOT USED. This function doesn't require the returned pixels to be peaks or summation peaks.
    #-----------------------------------------------------------------------------------------------
    def GetPeaks3(self, Threshold1, Threshold2, KnownPixels):

        Pixels = []
        PixelVals = []
        PixelSums = []

        # look through each pixel in KnownPixels
        for pixel in KnownPixels:
            (i, j, k) = pixel

            # for each neighbor of pixel
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    for kk in range(-1, 2):
                        if (ii, jj, kk) != (0, 0, 0):

                            # if neighbor-pixel-value > Threshold1
                            neighborPixel = (i+ii, j+jj, k+kk)
                            neighborPixelDensity = self.GetOneVal(i+ii, j+jj, k+kk, False)
                            if neighborPixelDensity > Threshold1:

                                # if neighbor-pixel-value > all its neighbors except pixel
                                if self.CenterPixelIsAlmostGreatestDensity(False, neighborPixel, pixel):

                                    # if neighbor's sum-of-neighbors > Threshold2
                                    (Answer, Sum) = self.CenterPixelIsGreatestDensity(False, i+ii, j+jj, k+kk)
                                    if Sum > Threshold2:

                                        # save this pixel
                                        Pixels.append(neighborPixel)
                                        PixelVals.append(neighborPixelDensity)
                                        PixelSums.append(Sum)

        return(Pixels, PixelVals, PixelSums)

    #---------------------------------------------------------------------------------------------------------
    # for each pixel in KnownPixels:
    #   look at all the pixel's nearest neighbors for:
    #     1. neighbor-pixel > Threshold1
    #     2. pixel-plus-neighbor-sums > pixel-plus-neighbor-sums of all its neighbors (i.e. a local max)
    # return: the neighbor pixel
    # return: the neighbor pixel-value
    # return: the sum of the neighbor-pixel and its neighbors
    # note: In other words, look for a local maximum in the sum of pixel-neighbors.
    #       Examine each neighbor of the pixels in KnownPixels
    #---------------------------------------------------------------------------------------------------------
    def GetPeaks4(self, Threshold1, KnownPixels):

        Pixels = []
        PixelVals = []
        PixelSums = []

        # look through each pixel in KnownPixels
        for pixel in KnownPixels:
            (i, j, k) = pixel

            # create a 5x5x5 array
            NeighborSums = np.zeros(shape=(5, 5, 5), dtype=float32)

            # get neighbor-sums for a 5x5x5 set of pixels, centered on a KnownPixel
            for ii in range(-2, 3):
                for jj in range(-2, 3):
                    for kk in range(-2, 3):
                        (Answer, Sum) = self.CenterPixelIsGreatestDensity(False, i+ii, j+jj, k+kk)
                        NeighborSums[ii+2, jj+2, kk+2] = Sum

            # look at the neighbor-sums in the 3x3x3 box that is centered in the 5x5x5 box
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    for kk in range(-1, 2):
                        # don't consider the center pixel because it's already a known pixel
                        if (ii==0 and jj==0 and kk==0): continue
                        # check if the neighbor-sum is a local max
                        CenterSum = NeighborSums[ii+2, jj+2, kk+2]
                        PixelVal = self.GetOneVal(i+ii, j+jj, k+kk, False)
                        # its corresponding pixel value must be > Threshold1
                        if PixelVal < Threshold1: continue
                        # check all the neighbor-sum's neighbors to see if it's a local max
                        IsMax = True
                        for iii in range(-1, 2):
                            if not IsMax: break
                            for jjj in range(-1, 2):
                                if not IsMax: break
                                for kkk in range(-1, 2):
                                    # don't compare sum to itself
                                    if (iii == 0 and jjj == 0 and kkk == 0): continue
                                    NeighborSum = NeighborSums[ii+2+iii, jj+2+jjj, kk+2+kkk]
                                    # 3x3x3 box pixel is a max if we can't find a neighbor that's bigger than it
                                    if NeighborSum > CenterSum:
                                        IsMax = False
                                        break
                        # if it's a max, save pixel info
                        if IsMax:
                            Pixels.append((i+ii,j+jj,k+kk))
                            PixelVals.append(PixelVal)
                            PixelSums.append(CenterSum)

        return(Pixels, PixelVals, PixelSums)

    # --------------------------------------------------------------------------------
    # return: True if the (i,j,k) pixel is greater than all its neighbors.
    # return: the sum of the (i,j,k) pixel + all its neighbors.
    # --------------------------------------------------------------------------------
    def CenterPixelIsGreatestDensity(self, InData, i, j, k):

        pixelDensity = self.GetOneVal(i, j, k, InData)
        IsCenterPixelLargestVal = True

        sum = pixelDensity
        for ii in range(-1, 2):
            for jj in range(-1, 2):
                for kk in range(-1, 2):
                    if (ii, jj, kk) != (0, 0, 0):
                        density = self.GetOneVal(i + ii, j + jj, k + kk, InData)
                        sum += density
                        if density > pixelDensity:
                            IsCenterPixelLargestVal = False

        if IsCenterPixelLargestVal:
            return(True, sum)
        else:
            return (False, sum)

    # --------------------------------------------------------------------------------
    # return: True if Pixel is greater than all its neighbors except ExcludePixel
    # return: the sum of Pixel and all its neighbors
    # --------------------------------------------------------------------------------
    def CenterPixelIsAlmostGreatestDensity(self, InData, Pixel, ExcludePixel):

        (i, j, k) = Pixel
        pixelDensity = self.GetOneVal(i, j, k, InData)
        IsCenterPixelAlmostLargestVal = True

        sum = pixelDensity
        for ii in range(-1, 2):
            for jj in range(-1, 2):
                for kk in range(-1, 2):
                    if (ii, jj, kk) != (0, 0, 0):
                        if (ii, jj, kk) != ExcludePixel:
                            density = self.GetOneVal(i+ii, j+jj, k+kk, InData)
                            sum += density
                            if density > pixelDensity:
                                IsCenterPixelAlmostLargestVal = False

        if IsCenterPixelAlmostLargestVal:
            return(True, sum)
        else:
            return(False, sum)

    # --------------------------------------------------------------------------------
    # from estimatedPositions pick a set of N positions that give the closest
    # match to the Patterson map on the NN input.
    # return the subset of atoms that are selected.
    # --------------------------------------------------------------------------------
    def PickBestSet(self, estimatedPositions, N, AtomRadiusSqr, NumSteps):

        DeltaStart = 0.001 * 2.0
        DeltaStop = 0.001 / 10.0
        DeltaInc = (DeltaStop - DeltaStart) / NumSteps
        Delta = DeltaStart

        # this is to see what happens with a greedy algorithm
        # it might work just as well this way, but I'm leaving simulated-annealing in place for now.
        # Delta = 0
        # DeltaInc = 0

        # pick N estimatedPositions at random.  this is the initial guess.
        WhichOnes = self.PickN(len(estimatedPositions), N)
        Subset = self.GetSubset(WhichOnes, estimatedPositions)
        Types = ['C'] * N

        # calculate a density map for the N random positions. this goes on the NN output.
        self.MakeMap(Subset, Types, False, AtomRadiusSqr, ClearData=True)

        # calculate a Patterson map for the density map on the NN output. this goes on the work array.
        self.MakePattersonMap(outputIsWorkArray=True)

        # get a similarity score (MSE) between the true Patterson map (NN input) and the test Patterson map (work array)
        SimilarityScore = self.ComparePattersonMaps()
        print("Similarity Score = %f" % (SimilarityScore))

        # repeat many times
        count = 0
        while SimilarityScore > 0.0001:

            # subtract off the density for one of the selected atoms, chosen randomly
            oldIndex = self.PickOne(True, WhichOnes)
            self.AddToDensity(estimatedPositions[oldIndex], False, AtomRadiusSqr, SubtractInstead=True)

            # add the density for one of the unselected atoms, chosen randomly
            newIndex = self.PickOne(False, WhichOnes)
            self.AddToDensity(estimatedPositions[newIndex], False, AtomRadiusSqr, SubtractInstead=False)

            # calculate the Patterson map for the new density map. this goes in the work array.
            self.MakePattersonMap(outputIsWorkArray=True)

            # get similarity score (MSE) between true Patterson map (NN input) and test Patterson map (work array)
            TestSimilarityScore = self.ComparePattersonMaps()

            # if the similarity score improved
            if TestSimilarityScore < (SimilarityScore + Delta):

                # keep the changes
                SimilarityScore = TestSimilarityScore
                WhichOnes[oldIndex] = False
                WhichOnes[newIndex] = True

                old = estimatedPositions[oldIndex]
                new = estimatedPositions[newIndex]
                print( "%4d:  old:%8.3f,%8.3f,%8.3f" %   (count, old[0], old[1], old[2]))
                print("       new:%8.3f,%8.3f,%8.3f -> %8.5f" % (new[0], new[1], new[2], SimilarityScore))

            else:

                # revert the changes
                self.AddToDensity(estimatedPositions[oldIndex], False, AtomRadiusSqr, SubtractInstead=False)
                self.AddToDensity(estimatedPositions[newIndex], False, AtomRadiusSqr, SubtractInstead=True)

            count += 1
            if count > NumSteps:
                break

            Delta += DeltaInc

        return(self.GetSubset(WhichOnes, estimatedPositions))

    # --------------------------------------------------------------------------------
    # pass: M the length of a new bool list.
    #       N the number of trues in the list
    # return: a list of bools of length M, with N set to true
    # --------------------------------------------------------------------------------
    def PickN(self, M, N):

        assert(N < M)

        WhichOnes = [False] * M
        count = 0
        while count < N:
            index = np.random.randint(0, M)
            if not WhichOnes[index]:
                WhichOnes[index] = True
                count += 1

        return(WhichOnes)

    # --------------------------------------------------------------------------------
    # if TrueOrFalse = True, pick one of the True items in WhichOnes
    # if TrueOrFalse = False, pick one of the False items in WhichOnes
    # return: the index of the item
    # --------------------------------------------------------------------------------
    def PickOne(self, TrueOrFalse, WhichOnes):
        while True:
            index = np.random.randint(0, len(WhichOnes))
            # if the condition is met, return the index
            if WhichOnes[index] == TrueOrFalse:
                return(index)

    # --------------------------------------------------------------------------------
    # return a subset of the coordinates in Positions.
    # WhichOnes is a bool list indicating which Positions to return.
    # --------------------------------------------------------------------------------
    def GetSubset(self, WhichOnes, Positions):
        assert(len(WhichOnes) == len(Positions))
        Subset = []
        for i in range(len(WhichOnes)):
            if WhichOnes[i]:
                Subset.append(Positions[i])
        return(Subset)
