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
        plot_list = [0] * 28
        for y in range(28):
            listx = [0] * 28
            for x in range(28):
                listx[x] = self.images[id][x + y * 28]
            plot_list[y] = listx

        print(self.labels[id])

        import matplotlib.pyplot as plt
        plt.imshow(plot_list, cmap="viridis", interpolation="nearest")
        try:
            plt.show()
        except KeyboardInterrupt:
            print("Plot has been externally interrupted")
            pass
