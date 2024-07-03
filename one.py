import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import rotate
import global_params
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

import torch
from torch import nn

from torch.nn import (
    Conv2d,
    Linear,
    ReLU,
    MaxPool2d,
    Flatten,
    Sequential,
)

size = global_params.Size


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64 * 16 * 16, 64 * 4),
            Linear(64 * 4, 14),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# define transform
transform = transforms.Compose(
    [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]
)


def single_test(path):
    roi = Image.open(path)
    roi = transform(roi)
    img = torch.reshape(roi, (1, 3, size, size))
    output = model(img)
    label = id_to_species[output.argmax(1).item()]
    plt.imshow(roi.permute(1, 2, 0).numpy())
    plt.title(label)
    plt.show()


def batch_test():
    for type in chess_pieces_types:
        path = f"./data/{type}/{type}_0.png"
        single_test(path)


chess_pieces_types = global_params.Chess_pieces_types

species_to_id = dict((c, i) for i, c in enumerate(chess_pieces_types))
print(species_to_id)
id_to_species = dict((v, k) for k, v in species_to_id.items())
print(id_to_species)


model = torch.load("models/model.pth")

path = f"./data/image.png"
# single_test(path)
batch_test()
