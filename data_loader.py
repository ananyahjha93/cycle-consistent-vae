import random
import numpy as np
from itertools import cycle
from utils import imshow_grid, transform_config

from torchvision import datasets
from torch.utils.data import Dataset, DataLoader


class MNIST_Paired(Dataset):
    def __init__(self, root='mnist', download=True, train=True, transform=transform_config):
        self.mnist = datasets.MNIST(root=root, download=download, train=train, transform=transform)

        self.data_dict = {}

        for i in range(self.__len__()):
            image, label = self.mnist.__getitem__(i)

            try:
                self.data_dict[label.item()]
            except KeyError:
                self.data_dict[label.item()] = []
            self.data_dict[label.item()].append(image)

    def __len__(self):
        return self.mnist.__len__()

    def __getitem__(self, index):
        image, label = self.mnist.__getitem__(index)

        # return another image of the same class randomly selected from the data dictionary
        # this is done to simulate pair-wise labeling of data
        return image, random.SystemRandom().choice(self.data_dict[label.item()]), label


if __name__ == '__main__':
    """
    test code for data loader
    """
    mnist_paired = MNIST_Paired()
    loader = cycle(DataLoader(mnist_paired, batch_size=16, shuffle=True, num_workers=0, drop_last=True))

    print(mnist_paired.data_dict.keys())

    image_batch, image_batch_2, labels_batch = next(loader)
    print(labels_batch)

    image_batch = np.transpose(image_batch, (0, 2, 3, 1))
    image_batch = np.concatenate((image_batch, image_batch, image_batch), axis=3)
    imshow_grid(image_batch)

    image_batch_2 = np.transpose(image_batch_2, (0, 2, 3, 1))
    image_batch_2 = np.concatenate((image_batch_2, image_batch_2, image_batch_2), axis=3)
    imshow_grid(image_batch_2)
