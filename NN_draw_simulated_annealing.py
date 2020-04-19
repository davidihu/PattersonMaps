# author: David Hurwitz
# started: 11/05/19
#
# Read in similarity scores calculated during simulated annealing optimizations.
# Similarity scores measure the difference between known and test Patterson maps.
#
# 10 of 20 atoms were selected at random. An atom not in the set was substituted
# for an atom in the set and a new similarity score was calculated. Improvements
# are accepted and worse scores were accepted according to an annealing schedule.

from matplotlib import pyplot

path = "C:/Users/david/Documents/newff/results/NN/Patterson_maps/figures/simulated_annealing/"
filename1 = "patterson_position_accuracy_t6.txt"
filename2 = "simulated_annealing_t6.png"

iterations = []
mulitple_iterations = []

scores = []
multiple_scores = []

# open the file
input = open(path + filename1, 'r')

# this state indicates we're skipping lines between optimizations
skipping = True

# read through the file
for line in input:

    # skip until we get to a "Similarity Score" line
    if skipping:
        if "Similarity Score" not in line:
            continue

        # handle the 1st similarity score of the optimization
        parts = line.split(' ')
        iterations.append(0)
        scores.append(float(parts[3]))
        skipping = False
        continue

    # get the iteration
    if "old:" in line:
        parts = line.split(':')
        iterations.append(int(parts[0])+1)
        continue

    # get the similarity score
    if "new:" in line:
        parts = line.split('>')
        scores.append(float(parts[1]))
        continue

    # save this set of iterations/scores
    mulitple_iterations.append(iterations.copy())
    multiple_scores.append(scores.copy())

    # move on to the next set
    iterations.clear()
    scores.clear()
    skipping = True

input.close()

fig = pyplot.figure(1, figsize=(12,8))
pyplot.plot(mulitple_iterations[0], multiple_scores[0], 'r-')
pyplot.plot(mulitple_iterations[1], multiple_scores[1], 'g-')
pyplot.plot(mulitple_iterations[2], multiple_scores[2], 'b-')
pyplot.plot(mulitple_iterations[3], multiple_scores[3], 'c-')
pyplot.plot(mulitple_iterations[4], multiple_scores[4], 'm-')
pyplot.plot(mulitple_iterations[5], multiple_scores[5], 'y-')
pyplot.plot(mulitple_iterations[6], multiple_scores[6], 'k-')
pyplot.title("Patterson Map Comparisons for Simulated Annealing Runs")
pyplot.xlabel("iterations", fontweight='bold')
pyplot.ylabel("mean-squared error", fontweight='bold')
pyplot.legend(('run1', 'run2', 'run3', 'run4', 'run5', 'run6', 'run7'))
pyplot.show()

pyplot.savefig(path + filename2)

test = 1