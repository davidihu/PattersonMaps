# author: David Hurwitz
# started: 10/21/19
#
# Read in histogram file with bins for each accuracy range.
# Make a plot of the data.

from matplotlib import pyplot

path = "C:/Users/david/Documents/newff/results/NN/Patterson_maps/figures/"
filename1 = "patterson_position_accuracy_t4.histogram.txt"
filename2 = "patterson_position_accuracy_t4.histogram.png"

# would be nice to add this to the plot somewhere
# "average accuracy = 0.283"

# read the data
bin_starts = []
bin_stops = []
bin_numbers = []
input = open(path + filename1, 'r')
count = 0
for line in input:
    if count == 0:
        count += 1
        continue
    parts = line.split(',')
    bin_starts.append(float(parts[0]))
    bin_stops.append(float(parts[1]))
    bin_numbers.append(int(parts[2]))
    count += 1
input.close()

# these are the labels for the x axis
labels = []
for i in range(len(bin_numbers)):
    if i%2 == 1:
        label = ""
    else:
        label = "%4.2f" % (bin_starts[i])
    labels.append(label)

Title = "Positional accuracy of 5000 atoms"

# plot the histogram
pyplot.figure(1, figsize=(9,6))
pyplot.bar(range(len(bin_numbers)), bin_numbers, tick_label=labels, align='edge', width=1.0, edgecolor='k')
pyplot.title(Title)
pyplot.xlabel("pixels")
pyplot.xticks(rotation=60)
pyplot.ylabel("bin size", rotation='vertical')
pyplot.text(26, 900, "average accuracy = 0.283 pixels")
pyplot.show()

# save to file
savefile = path + filename2
pyplot.savefig(savefile)

test = 1