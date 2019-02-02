import matplotlib.pyplot as plt
import numpy as np

data = np.load("../Mat/train_x.npy")
for i in range(len(data)):
    if ((i % 30) == 0):
        plt.plot(data[i])
        plt.ylim((-600, 250))
        plt.savefig("../plots accent/"+str(i) + ".png", dpi=1500)
