
from qampy import signals, impairments
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from tensorboardX import SummaryWriter


def plot_constellation(E):
    plt.scatter(E[0].real, E[0].imag, alpha=0.4, color='yellow', edgecolors='black')
    plt.scatter(E[1].real, E[1].imag, alpha=0.4, color='blueviolet', edgecolors='black')
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.show()


def plot_clusters(sig_gauss, labels):
    for label_name in label_names.values():
        plt.scatter(sig_gauss[labels == label_name].real, sig_gauss[labels == label_name].imag, alpha=0.4)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.show()


def plot_results(data, labels, i):
    for label_name in sorted(label_names.values()):
        if torch.sum(labels == label_name) > 0:
            plt.scatter(data[labels == label_name][:, 0], data[labels == label_name][:, 1])
        else:
            plt.scatter(0,0,alpha=0)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.savefig(os.path.join("output", "{}.png".format(i)))
    plt.clf()

writer = SummaryWriter('summaries/linear')

sig = signals.SignalQAMGrayCoded(64, 2**20, fb=25e9, nmodes=2)
sig_gauss = impairments.change_snr(sig, 15.)
sig = np.array(sig)
label_names = {key: value for (key, value) in zip(np.unique(sig), range(len(np.unique(sig))))}

plot_constellation(sig_gauss)

sig = np.hstack((sig[0], sig[1]))
sig_gauss = np.hstack((sig_gauss[0], sig_gauss[1]))

labels = np.array([label_names[label] for label in sig])

label_tensor = torch.Tensor(labels)
signal_tensor = torch.Tensor([sig_gauss.real, sig_gauss.imag]).permute(1,0)

plot_clusters(sig_gauss, labels)

signal_batches = torch.split(signal_tensor, 64)
label_batches = torch.split(label_tensor, 64)

split_point = int(len(signal_batches)*0.75)

signal_training = signal_batches[:split_point]
label_training = label_batches[:split_point]
signal_testing = signal_batches[split_point:]
label_testing = label_batches[split_point:]


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 64),
        )

    def forward(self, input):
        return self.layers(input)


net = Network()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

lspace = torch.linspace(-1.5, 1.5, steps=100)
X, Y = np.meshgrid(lspace, lspace)
plane = torch.Tensor([X.flatten(), Y.flatten()]).to(torch.float).t()

for i, (data, label) in enumerate(zip(signal_training, label_training)):
    pred = net(data)
    optimizer.zero_grad()
    loss = criterion(pred, label.long())
    loss.backward()
    optimizer.step()
    accuracy = torch.sum(label == torch.argmax(pred, 1).float()).float()/label.size()[0]
    writer.add_scalar("accuracy", accuracy, i)
    writer.add_scalar("loss", loss, i)

    if i % 10 == 0:
        print('Step {:04d}, loss {:.6f}'.format(i, loss))

        plane_result = net(plane)
        plot_results(plane, torch.argmax(plane_result, 1), i)


data = torch.stack(signal_testing).view(-1, 2)
predicted_labels = net(data)

label = torch.stack(label_testing).view(-1)

correct_pred = torch.sum(label == torch.argmax(predicted_labels,1).float()).float()
total_pred = label.size()[0]

plot_results(data, torch.argmax(predicted_labels, 1), "a")
print('\nAccuracy: {:.6f}'.format(correct_pred/total_pred))

