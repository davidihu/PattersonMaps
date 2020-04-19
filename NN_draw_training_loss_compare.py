# author: David Hurwitz
# started: 10/21/19
#
# Read in files that tracked training and validation loss.
# Plot the training and validation loss for multiple files in 1 plot.

from matplotlib import pyplot
from matplotlib.ticker import FormatStrFormatter

# the base path where the files are located
base_path = "C:/Users/david/Documents/newff/results/NN/Patterson_maps/NN_models/"

# the template for each output file name
data_files = "model_%02d/model_%02d_training_%02d.txt"

# the saved image goes here
out_path = "C:/Users/david/Documents/newff/results/NN/Patterson_maps/figures/"
out_file = out_path + "training_and_validation_loss_with_and_without_centrosymmetric_atoms.png"
# out_file = out_path + "training_and_validation_loss_for_3_different_inner_box_sizes.png"

# these are the colors for the centro-symmetry plot
colors = ['red', 'blue', 'magenta', 'deepskyblue']

# these are the colors for the inner-box plot
# colors = ['red', 'blue', 'green', 'darkorange']

model_training_list = []

# for centrosymmetric atoms test
model_training_list.append((20, 1))   # 10 original atoms + 10 centro-symmetric atoms
model_training_list.append((23, 1))   # 10 original atoms + 10 centro-symmetric atoms
model_training_list.append((24, 1))   # 10 original atoms + no centro-symmetric atoms
model_training_list.append((25, 1))   # 20 original atoms + no centro-symmetric atoms

# for box comparison test
# model_training_list.append((20, 1))   # inner box = [-7.125 : 7.125]
# model_training_list.append((23, 1))   # inner box = [-7.125 : 7.125]
# model_training_list.append((21, 1))   # inner box = [ -10.0 :  10.0]
# model_training_list.append((22, 1))   # inner box = [ -13.0 :  13.0]

# Xs and losses for one model/training run
Xs = []
TrainingLosses = []
ValidationLosses = []

# Xs and losses for all model/training runs
AllXs = []
AllTrainingLosses = []
AllValidationLosses = []

# for each model/training pair
for (model, training) in model_training_list:

    # Xs and losses for one model/training run
    x = 0
    Xs.clear()
    TrainingLosses.clear()
    ValidationLosses.clear()

    # open the file
    file = base_path + data_files % (model, model, training)
    input = open(file, 'r')

    # read through the file
    TrainingLoss = "unknown"
    ValidationLoss = "unknown"
    for line in input:

        # get training loss
        if ": loss:" in line:
            parts1 = line.split('[')
            parts2 = parts1[1].split(']')
            TrainingLoss = float(parts2[0])

        # get validation loss
        if "validation loss" in line:
            parts = line.split('=')
            ValidationLoss = float(parts[1])

            # show epochs on x-axis
            x += 1

            # handle the case when there are 10 atoms in the output map instead of 20
            if model == 24:
                TrainingLoss *= 2.0
                ValidationLoss *= 2.0

            Xs.append(x)
            TrainingLosses.append(TrainingLoss)
            ValidationLosses.append(ValidationLoss)
            assert((TrainingLoss != 'unknown') and (ValidationLoss != 'unknown'))

    input.close()

    AllXs.append(Xs.copy())
    AllTrainingLosses.append(TrainingLosses.copy())
    AllValidationLosses.append(ValidationLosses.copy())

# plot training and validation loss for each epoch
fig = pyplot.figure(1, figsize=(12,8))
ax = fig.add_subplot(111)
ax.yaxis.set_major_formatter(FormatStrFormatter('%4.2e'))
axes = pyplot.gca()
axes.set_ylim([0, 1.8E-4])
for i in range(0, len(AllXs)):
    pyplot.plot(Xs, AllTrainingLosses[i], color = colors[i], linestyle = '-')
    pyplot.plot(Xs, AllValidationLosses[i], color = colors[i], linestyle = (0,(2,1)))

pyplot.title("Training and Validation Loss, With and Without Centrosymmetric Atoms")
# pyplot.title("Training and Validation Loss, For Different Inner Box Sizes")

pyplot.xlabel("epochs", fontweight='bold')
pyplot.ylabel("mean squared error", fontweight='bold')
legend1 = 'training:     10 random + 10 centrosymmetric '
legend2 = 'validation:  10 random + 10 centrosymmetric '
legend5 = 'training:     10 random +   0 centrosymmetric  (scale loss)'
legend6 = 'validation:  10 random +   0 centrosymmetric  (scale loss)'
legend7 = 'training:     20 random +   0 centrosymmetric '
legend8 = 'validation:  20 random +   0 centrosymmetric '
# legend1 = 'training:     [-6 : 6]'
# legend2 = 'validation:  [-6 : 6]'
# legend5 = 'training:     [-9 : 9]'
# legend6 = 'validation:  [-9 : 9]'
# legend7 = 'training:     [-12 : 12]'
# legend8 = 'validation:  [-12 : 12]'
pyplot.legend((legend1, legend2, legend1, legend2, legend5, legend6, legend7, legend8))
pyplot.show()

# save plot 1 to file
pyplot.savefig(out_file)

test=1
