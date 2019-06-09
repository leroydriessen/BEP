from qampy import signals, impairments

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from lstm import LSTM
from constants import *
from functions import *

import numpy as np
from datetime import datetime
import random

print("\nStart, current time is "+datetime.now().strftime("%H:%M:%S"))

if not USE_REAL_DATA:
    print("Generating 2^{:d} random bits per polarization in {:d}-QAM format...".format(int(np.log2(AMOUNT_OF_SYMBOLS)), MODULATION_SIZE))

    # Generate random signal and keep adding glass fiber impairments
    sig_original = signals.SignalQAMGrayCoded(MODULATION_SIZE, AMOUNT_OF_SYMBOLS, fb=SYMBOL_RATE, nmodes=2)
    sig = sig_original

    if PLOT_PICTURES:
        plot_constellation(sig, "Signal constellation without distortions\n $F_{symbol}=" + "{:d}GBd$, #symbols=2^{:d}".format(int(SYMBOL_RATE / 1e9), int(np.log2(AMOUNT_OF_SYMBOLS))), "sig")
        plot_time(sig, "X-polarization over time without\n distortions $F_{symbol}=" + "{:d}GBd$".format(int(SYMBOL_RATE/1e9)), "sig_time")

    if USE_PULSESHAPING:
        print("Pulse-shaping the signal...")
        sig = sig.resample(OVER_SAMPLING*sig.fb, renormalise=True, beta=BETA)
        if PLOT_PICTURES:
            plot_constellation(sig, "Signal constellation after RRcos\n pulseshaping " + r"$\beta={:.1f}$, $F_s={:d}GHz$".format(BETA, int(SYMBOL_RATE*OVER_SAMPLING/1e9)), "sig_shaped")
            plot_time(sig, "X-polarization over time after RRcos\n pulseshaping, " + r"$\beta={:.1f}$, $F_s={:d}GHz$".format(BETA, int(SYMBOL_RATE*OVER_SAMPLING/1e9)), "sig_shaped_time")

    if USE_AGWN or USE_PMD or USE_PHASE_NOISE or USE_FREQ_OFFSET:
        print("Adding artificial signal impairments...")

    if USE_AGWN:
        sig = impairments.change_snr(sig, SNR)
        if PLOT_PICTURES:
            plot_constellation(sig, "Signal constellation after pulseshaping\n and quantum noise, $SNR={:d}$".format(SNR), "sig_agwn")
            plot_time(sig, "X-polarization over time after pulseshaping\n and quantum noise, $SNR={:d}$".format(SNR), "sig_agwn_time")

    if USE_PMD:
        sig = impairments.apply_PMD(sig, THETA, TDGD)
        if PLOT_PICTURES:
            plot_constellation(sig, "Signal constellation after pulseshaping, AGWN\n and PMD," + r"$\theta=\pi/{:d}$, $\Delta\tau ={:d}ps$".format(int((THETA/np.pi)**-1), int(TDGD*1e12)), "sig_agwn_pmd")
            plot_time(sig, "X-polarization over time after pulseshaping,\n AGWN and PMD,"+r"$\theta=\pi/{:d}$, $\Delta\tau ={:d}ps$".format(int((THETA/np.pi)**-1), int(TDGD*1e12)), "sig_agwn_pmd_time")

    if USE_PHASE_NOISE:
        sig = impairments.apply_phase_noise(sig, LINEWIDTH)
        if PLOT_PICTURES:
            plot_constellation(sig, "Signal constellation after pulseshaping, AGWN\n PMD and phase noise, linewidth of laser = {:d}MHz".format(int(LINEWIDTH/1e6)), "sig_agwn_pmd_phase")
            plot_time(sig, "Signal constellation after pulseshaping, AGWN\n PMD and phase noise, linewidth of laser = {:d}MHz".format(int(LINEWIDTH/1e6)), "sig_agwn_pmd_phase_time")

    if USE_FREQ_OFFSET:
        sig = impairments.add_carrier_offset(sig, FREQ_OFFSET)
        if PLOT_PICTURES:
            plot_constellation(sig, "Signal constellation at receiver, with AGWN, PMD,\n phase noise and frequency offset, $f_{offset}="+"{:d}MHz$".format(int(FREQ_OFFSET/1e6)), "sig_agwn_pmd_phase_freq")
            plot_time(sig, "Signal constellation at receiver, with AGWN, PMD,\n phase noise and frequency offset, $f_{offset}="+"{:d}MHz$".format(int(FREQ_OFFSET/1e6)), "sig_agwn_pmd_phase_freq_time")

    # lock the random seed to add traceability
    random.seed(420691337)


    # select the signal to use
    if DISREGARD_OVERSAMPLING and USE_PULSESHAPING:
        input_sig = sig[:, ::OVER_SAMPLING]
    else:
        input_sig = sig
else:
    real_data = np.load("8QAM_01loops_1.npy", allow_pickle=True)[()]
    sig_original = real_data["transmitted_symbols"]
    input_sig = real_data["received_samples"][:, :2**17]

    MODULATION_SIZE = 2**3
    OVER_SAMPLING = 2
    AMOUNT_OF_SYMBOLS = 2**16
    SYMBOL_RATE = 41.80e9
    BETA = 0.01
    DISREGARD_OVERSAMPLING = False

    if PLOT_PICTURES:
        plot_constellation(sig_original, "Real world 8-QAM data input constellation\nsymbol rate=41.80GBd, #symbols=2^16", "real_input")
        plot_constellation(input_sig[:, ::2], "Real world 8-QAM data output constellation\n(frequency offset compensated)", "real_output")


print("Manipulating data into 4d tensor...")

input_array = torch.Tensor(np.vstack((input_sig[0].real, input_sig[1].real, input_sig[0].imag, input_sig[1].imag)))

indices = list(range(AMOUNT_OF_SYMBOLS-SEQUENCE_LENGTH+1))  # AMOUNT_OF_SYMBOLS long, but subtracting the SEQUENCE LENGTH to avoid getting out of array bounds
if SHUFFLE:
    random.shuffle(indices)

# create data batches
if DISREGARD_OVERSAMPLING or not USE_PULSESHAPING:
    extracted_sequences = torch.stack([torch.t(torch.narrow(input_array, 1, i, SEQUENCE_LENGTH)) for i in indices])
else:
    extracted_sequences = torch.stack([torch.t(torch.narrow(input_array, 1, OVER_SAMPLING*i+1, SEQUENCE_LENGTH)) for i in indices])

batches = torch.stack(torch.split(extracted_sequences, BATCH_SIZE)[:-1])  # remove the last batch since it might be of smaller dimension than the rest of the batches
input_batches = batches.permute(0, 2, 1, 3)  # permute to account for expected LSTM input

print("Generating correct labels...")

remove_index = int(SEQUENCE_LENGTH/2) if not DISREGARD_OVERSAMPLING and USE_PULSESHAPING else SEQUENCE_LENGTH-1

if not USE_REAL_DATA:
    input_labels = sig_original.bits
else:
    input_labels = [[], []]
    for j in range(2):
        for i in range(len(sig_original[0])):
            if sig_original[j][i] == -1.+1.j:
                input_labels[j].extend([False, True, True])
            elif sig_original[j][i] == 0.7320508075688772j:
                input_labels[j].extend([True, True, True])
            elif sig_original[j][i] == 1.+1.j:
                input_labels[j].extend([True, False, True])
            elif sig_original[j][i] == -0.7320508075688772:
                input_labels[j].extend([False, False, True])
            elif sig_original[j][i] == 0.7320508075688772:
                input_labels[j].extend([True, False, False])
            elif sig_original[j][i] == -1.-1.j:
                input_labels[j].extend([False, False, False])
            elif sig_original[j][i] == -0.7320508075688772j:
                input_labels[j].extend([False, True, False])
            elif sig_original[j][i] == 1.-1.j:
                input_labels[j].extend([True, True, False])
            else:
                exit(1)
    input_labels = np.array(input_labels)

# create label batches
if SHUFFLE:
    labels = []
    labels_ordered = np.array(input_labels.transpose().reshape(-1, int(np.log2(MODULATION_SIZE)*2)).astype(int)[remove_index:])  # remove the first SEQUENCE_LENGTH-1 labels, as there do not exist data sequences that use those labels
    for i in indices:
        labels.append(labels_ordered[i])
    labels = torch.Tensor(labels)
else:
    labels = torch.Tensor(np.array(input_labels.transpose().reshape(-1, int(np.log2(MODULATION_SIZE)*2)).astype(int))[remove_index:])  # remove the first SEQUENCE_LENGTH-1 labels, as there do not exist data sequences that use those labels
label_batches = torch.stack(torch.split(labels, BATCH_SIZE)[:-1])  # remove the last batch since it might be of smaller dimension than the rest of the batches

# create network, optimizer and loss function
print("Initializing neural network...")

lstm = LSTM(input_dim=4, num_hidden=HIDDEN_NODES, hidden_layers=HIDDEN_LAYERS, num_classes=int(np.log2(MODULATION_SIZE)*2), batch_size=BATCH_SIZE, device="cpu")
optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# initiate summary writer
if TENSORBOARD:
    directory = './summaries/'+datetime.now().strftime("%Y-%m-%d_%H:%M")+"_mod{:d}_snr{:d}_lw{:.0E}_freq%{:.0E}_seq{:d}_lr{:.4f}_nn{:d}x{:d}_bs{:d}".format(MODULATION_SIZE, SNR, LINEWIDTH, FREQ_OFFSET, SEQUENCE_LENGTH, LEARNING_RATE, HIDDEN_NODES, HIDDEN_LAYERS, BATCH_SIZE)
    print("Initializing Tensorboard at " + directory + "...")
    writer = SummaryWriter(directory)

# split the batches into a training and testing set of batches
split_point = int(input_batches.size()[0]*TRAINING_RATIO)
input_training = input_batches[:split_point]
label_training = label_batches[:split_point]
input_testing = input_batches[split_point:]
label_testing = label_batches[split_point:]

print("Training the network...")

# train the LSTM
for step, (data, label) in enumerate(zip(input_training, label_training)):
    # set the optimizer gradient to zero
    optimizer.zero_grad()

    # push data batch through the network and get the prediction of the final
    prediction = lstm(data)

    # calculate the MSE error
    loss = criterion(prediction, label)

    # do backpropagation and update weights
    loss.backward()
    optimizer.step()

    # calculate total accuracy (BER) of current batch
    ber = float(torch.sum(torch.where(prediction > 0.5, torch.ones(BATCH_SIZE, int(np.log2(MODULATION_SIZE)*2)), torch.zeros(BATCH_SIZE, int(np.log2(MODULATION_SIZE)*2))) != label)) / float(BATCH_SIZE * int(np.log2(MODULATION_SIZE)*2))
    accuracy = 100*float(torch.sum(torch.where(prediction > 0.5, torch.ones(BATCH_SIZE, int(np.log2(MODULATION_SIZE)*2)), torch.zeros(BATCH_SIZE, int(np.log2(MODULATION_SIZE)*2))) == label)) / float(BATCH_SIZE * int(np.log2(MODULATION_SIZE)*2))
    if step % 20 == 0: #step % 100 == 0 or step < 10 or step < 100 and step % 10 == 0:
        if BER:
            print("Step {:04d}/{:04d}:\t{:.2E} BER".format(step, input_training.size()[0], ber))
        else:
            print("Step {:04d}/{:04d}:\t{:2.6f}% accuracy".format(step, input_training.size()[0], accuracy))

    if TENSORBOARD:
        writer.add_scalar("loss", loss, step)
        writer.add_scalar("BER", ber, step)
        writer.add_scalar("accuracy", accuracy, step)

print("Finished training, starting testing phase...")

correct = 0

for (data, label) in zip(input_testing, label_testing):
    optimizer.zero_grad()
    prediction = lstm(data)
    loss = criterion(prediction, label)
    correct += float(torch.sum(torch.where(prediction > 0.5, torch.ones(BATCH_SIZE, int(np.log2(MODULATION_SIZE)*2)), torch.zeros(BATCH_SIZE, int(np.log2(MODULATION_SIZE)*2))) == label))

print("Final accuracy with test data: {:.6f}".format(correct/(input_testing.size()[0]*BATCH_SIZE*int(np.log2(MODULATION_SIZE)*2))))

print("End, finishing time: " + datetime.now().strftime("%H:%M:%S"))
