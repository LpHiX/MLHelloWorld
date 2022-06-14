import random
from mnist import MNIST


class MNISTReader:
    mndata = None

    def __init__(self):
        self.mndata = MNIST('samples')
        self.images, self.labels = MNIST('samples').load_training()

    def return_set(self, id):
        return self.images[id], self.labels[id]

    def plot_data(self, id):
        list = [0] * 28
        for y in range(28):
            listx = [0] * 28
            for x in range(28):
                listx[x] = self.images[id][x + y * 28]
            list[y] = listx

        print(self.labels[id])

        import matplotlib.pyplot as plt
        plt.imshow(list, cmap="viridis", interpolation="nearest")
        plt.show()
