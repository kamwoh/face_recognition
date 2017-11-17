#!/usr/bin/env python

import numpy as np
import time
import matplotlib
from matplotlib import pyplot as plt


def getFigAx():
    fig, ax = plt.subplots(1, 1)
    ax.hold(True)
    plt.show(False)
    plt.draw()
    return fig, ax


def closeFig(fig):
    plt.close(fig)


def refresh(fig, axes, Y, color):
    # for x, y in enumerate(Y):
    axes.plot(np.arange(len(Y)), Y, color)
    fig.canvas.draw()
