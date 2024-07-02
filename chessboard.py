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


chess_pieces = global_params.Chess_pieces

# image = cv2.imread("chess.jpg")
image = cv2.imread("chess.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(
    img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=50
)

size = global_params.Size


chess_pieces_types = global_params.Chess_pieces_types

species_to_id = dict((c, i) for i, c in enumerate(chess_pieces_types))
print(species_to_id)
id_to_species = dict((v, k) for k, v in species_to_id.items())
print(id_to_species)


# define transform
transform = transforms.Compose(
    [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ]
)


# load model
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


model = torch.load("models/model.pth")
print(model)


# If some circles are detected, separate each chess piece
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i, circle in enumerate(circles[0, :]):
        # Get the circle parameters
        x, y, r = circle
        # Extract the region of interest (ROI) containing the chess piece
        roi = image[y - r : y + r, x - r : x + r]

        # resize the image
        roi = cv2.resize(roi, (size, size))

        # roi = np.transpose(roi, (2, 0, 1))
        # roi = torch.tensor(roi, dtype=torch.float32) / 255.0
        # roi = torch.reshape(roi, (1, 3, size, size))
        cv2.imwrite("./data/image.png", roi)
        roi = Image.open(f"./data/image.png")
        roi = transform(roi)
        img = torch.reshape(roi, (1, 3, size, size))
        output = model(img)
        label = id_to_species[output.argmax(1).item()]
        plt.imshow(roi.permute(1, 2, 0).numpy())
        plt.title(label)
        plt.show()


# Display the original image with detected circles
for i in circles[0, :]:
    cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), -1)

# Show the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
