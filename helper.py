import numpy as np
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, time_taken, std_dev_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    # Plotting scores
    plt.subplot(3, 1, 1)
    plt.title('Satitistics')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    # Plotting time taken
    plt.subplot(3, 1, 2)
    plt.xlabel('Number of Games')
    plt.ylabel('Time Survived')
    plt.plot(time_taken)

    # Plotting standard deviation
    plt.subplot(3, 1, 3)
    plt.plot(std_dev_scores, marker='o', linestyle='-')
    plt.xlabel('Number of Games')
    plt.ylabel('Standard Deviation')
    plt.grid(True)
    plt.show()

    plt.show(block=False)
    plt.pause(.1)
