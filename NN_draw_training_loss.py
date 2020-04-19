# author: David Hurwitz
# started: 10/21/19
#
# Read in files that tracked training and validation loss.
# Plot the training and validation loss across multiple files.

from matplotlib import pyplot
from matplotlib.ticker import FormatStrFormatter

# the base path where the files are located
base_path = "C:/Users/david/Documents/newff/results/NN/Patterson_maps/NN_models/"

# the template for each output file name
data_files = "model_%02d/model_%02d_training_%02d.txt"

# the saved image goes here
out_path = "C:/Users/david/Documents/newff/results/NN/Patterson_maps/figures/"
# out_file1 = out_path + "training_and_validation_loss_initial_part_all_epochs.png"
out_file1 = out_path + "training_and_validation_loss_full_training_all_epochs.png"
out_file2 = out_path + "training_and_validation_loss_full_training_last_epochs.png"

# for box comparison test
model_training_list = []
model_training_list.append((11, 1))
model_training_list.append((12, 1))

# all the (model, training) pairs: 1/(86-90) & 8/(1-21)
# model = 1
# for training in range(86, 91):
#     model_training_list.append((model, training))
# model = 8
# for training in range(1, 22):
#     model_training_list.append((model, training))

Xs = []
TrainingLosses = []
ValidationLosses = []

XsFirstEpoch = []
TrainingLossesFirstEpoch = []
ValidationLossesFirstEpoch = []

XsLastEpoch = []
TrainingLossesLastEpoch = []
ValidationLossesLastEpoch = []

# for plotting just the initial part of the curves
TruncateAt = 300
# TruncateAt = -1

# for each model/training pair
x = 0
for (model, training) in model_training_list:

    # open the file
    file = base_path + data_files % (model, model, training)
    input = open(file, 'r')

    # read through the file
    TrainingBatchSize = NumEpochsPerMakeNewData = "unknown"
    TrainingLoss = ValidationLoss = "unknown"
    MakeNewDataCount = 0
    for line in input:

        # get training batch size
        if "TrainingBatchSize" in line:
            parts = line.split(" ")
            TrainingBatchSize = int(parts[1])

        # get number-of-epochs-per-make-new-data
        if "NumEpochsPerMakeNewData" in line:
            parts = line.split(" ")
            NumEpochsPerMakeNewData = int(parts[1])

        # get training loss
        if ": loss:" in line:
            parts1 = line.split('[')
            parts2 = parts1[1].split(']')
            TrainingLoss = float(parts2[0])

        # get validation loss
        if "validation loss" in line:
            parts = line.split('=')
            ValidationLoss = float(parts[1])

            # save the batch size and losses for display later
            # x += TrainingBatchSize     # do it this way for (epochs * batch-size) on x-axis
            x += 1                       # do it this way for epochs on x-axis

            # check if we're truncating the input data
            if (TruncateAt > 0) and (x >= TruncateAt):
                break

            Xs.append(x)
            TrainingLosses.append(TrainingLoss)
            ValidationLosses.append(ValidationLoss)
            assert((TrainingBatchSize != 'unknown') and (TrainingLoss != 'unknown') and (ValidationLoss != 'unknown'))

            if MakeNewDataCount == 0:
                XsFirstEpoch.append(x)
                TrainingLossesFirstEpoch.append(TrainingLoss)
                ValidationLossesFirstEpoch.append(ValidationLoss)

            if MakeNewDataCount == NumEpochsPerMakeNewData-1:
                XsLastEpoch.append(x)
                TrainingLossesLastEpoch.append(TrainingLoss)
                ValidationLossesLastEpoch.append(ValidationLoss)

            MakeNewDataCount += 1
            if MakeNewDataCount == NumEpochsPerMakeNewData:
                MakeNewDataCount = 0

    input.close()

# plot training and validation loss for each epoch
fig = pyplot.figure(1, figsize=(12,8))
ax = fig.add_subplot(111)
ax.yaxis.set_major_formatter(FormatStrFormatter('%4.2e'))
pyplot.plot(Xs, TrainingLosses, 'r-')
pyplot.plot(Xs, ValidationLosses, 'b-')
pyplot.title("Training and Validation Loss")
pyplot.xlabel("epochs", fontweight='bold')
pyplot.ylabel("mean squared error", fontweight='bold')
pyplot.legend(('training', 'validation'))
pyplot.show()

# save plot 1 to file
pyplot.savefig(out_file1)

# plot training and validation loss for the first epoch of each NumEpochsPerMakeNewData
fig = pyplot.figure(2)
ax = fig.add_subplot(111)
ax.yaxis.set_major_formatter(FormatStrFormatter('%4.2e'))
pyplot.plot(XsFirstEpoch, TrainingLossesFirstEpoch, 'r-')
pyplot.plot(XsFirstEpoch, ValidationLossesFirstEpoch, 'b-')
pyplot.title("Training and Validation Loss")
pyplot.xlabel("epochs", fontweight='bold')
pyplot.ylabel("mean squared error", fontweight='bold')
pyplot.legend(('training', 'validation'))
pyplot.show()

# plot training and validation loss for the last epoch of each NumEpochsPerMakeNewData
fig = pyplot.figure(3, figsize=(12,8))
ax = fig.add_subplot(111)
ax.yaxis.set_major_formatter(FormatStrFormatter('%4.2e'))
pyplot.plot(XsLastEpoch, TrainingLossesLastEpoch, 'r-')
pyplot.plot(XsLastEpoch, ValidationLossesLastEpoch, 'b-')
pyplot.title("Training and Validation Loss")
pyplot.xlabel("epochs", fontweight='bold')
pyplot.ylabel("mean squared error", fontweight='bold')
pyplot.legend(('training', 'validation'))
pyplot.show()

# save plot 3 to file
pyplot.savefig(out_file2)

test=1
