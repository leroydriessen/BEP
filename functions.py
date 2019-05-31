import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from qampy import signals

"""
Plot the constellation of a dual polarized signal
Parameters:
  E: data sequence to plot
  T: title of graph
"""
def plot_constellation(E, T, F):
    if isinstance(E, signals.SignalQAMGrayCoded):
        os = int(E.fs / E.fb)
    else:
        os = 1
    plt.scatter(E[0].real[::os], E[0].imag[::os], alpha=0.4, color='yellow', edgecolors='black', label="X-polarization")
    plt.scatter(E[1].real[::os], E[1].imag[::os], alpha=0.4, color='blueviolet', edgecolors='black',
                label="Y-polarization")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title(T)
    plt.legend(loc=3)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.savefig(F+".png")
    plt.figure(figsize=(5, 5))
    plt.show()

'''
Plot the signal over time in 3D, if the signal has more than 16 symbols, clip it to the first 16 symbols
Only plots X-polarization
Axes:
  x: symbol time
  y: quadrature component of the symbol of the X-polarization
  z: in-phase component of the symbol of the X-polarization
Parameters:
  E: data sequence to plot
  T: title of graph
'''
def plot_time(E, T, F):
    os = int(E.fs / E.fb)
    if E.shape[1] / os > 16:
        E = E[0:16 * os, 0:16 * os]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca(projection='3d')
    ax.plot(np.linspace(0, E.shape[1] / (E.fs / E.fb), E.shape[1], endpoint=False), E[0].real, zdir='y', label=r'$X_I$')
    ax.plot(np.linspace(0, E.shape[1] / (E.fs / E.fb), E.shape[1], endpoint=False), E[0].imag, zdir='z', label=r'$X_Q$')
    ax.add_collection3d(
        plt.fill_between(np.linspace(0, E.shape[1] / (E.fs / E.fb), E.shape[1], endpoint=False), E[0].real, 0,
                         alpha=0.3), 0, zdir='y')
    ax.add_collection3d(
        plt.fill_between(np.linspace(0, E.shape[1] / (E.fs / E.fb), E.shape[1], endpoint=False), E[0].imag, 0,
                         alpha=0.3), 0, zdir='z')
    ax.view_init(elev=40., azim=-65)
    ax.set_xlabel("Time")
    ax.set_ylabel("In-phase")
    ax.set_zlabel("Quadrature")
    ax.set_xlim([0, 16])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title(T, pad=20)
    ax.legend(loc=6)
    plt.savefig(F+".png")
    plt.show()
