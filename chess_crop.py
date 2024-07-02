import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import rotate
import os
import global_params
import random
from eprogress import LineProgress

line_progress = LineProgress(title="line progress")


chess_pieces = global_params.Chess_pieces


for chess in chess_pieces:
    path = f"./data/{chess}"
    if not os.path.exists(path):
        os.makedirs(path)


def check_chess_piece(roi, i):
    """
    Check if the extracted chess piece is correct
    """
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.title(chess_pieces[i])
    plt.show()


image = cv2.imread("chess.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(
    img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=50
)

size = 128


def rotate_image(image):
    size = image.shape[0]
    colors = np.array(
        [
            image[2, size - 2],
            image[2, size - 2],
            image[size - 2, 2],
            image[size - 2, size - 2],
        ]
    )
    angle = random.randint(1, 360)
    rotated_image = rotate(image, angle, reshape=False)
    for k in range(size):
        for j in range(size):
            if (rotated_image[k, j] == [0, 0, 0]).all():
                rotated_image[k, j] = random.choice(colors)
    return rotated_image


def bright_image(image):
    # bright_contrast = []
    alpha = random.uniform(0.5, 2)
    bright_image = cv2.convertScaleAbs(image, alpha=alpha)
    return bright_image


def contrast_image(image):
    beta = random.uniform(-100, 100)
    contrast_image = cv2.convertScaleAbs(image, beta=beta)
    return contrast_image


def noisy_image(image):
    noise = np.random.normal(0, 25, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


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

        # Save the ROI as a separate image
        cnt = 0
        for iter in range(360):
            tmp_img = roi  # .copy()

            # ROTATE IMAGE
            # tmp_img = rotate(tmp_img, iter, reshape=False)
            tmp_img = rotate_image(tmp_img)
            cv2.imwrite(
                f"./data/{chess_pieces[i]}/{chess_pieces[i]}_{cnt}.png", tmp_img
            )
            cnt += 1

            # BRIGHTNESS
            if random.randint(0, 1):
                tmp_img = bright_image(tmp_img)
                cv2.imwrite(
                    f"./data/{chess_pieces[i]}/{chess_pieces[i]}_{cnt}.png", tmp_img
                )
                cnt += 1
            # CONTRAST
            if random.randint(0, 1):
                tmp_img = contrast_image(tmp_img)
                cv2.imwrite(
                    f"./data/{chess_pieces[i]}/{chess_pieces[i]}_{cnt}.png", tmp_img
                )
                cnt += 1

            # ADD NOISE
            if random.randint(0, 1):
                tmp_img = noisy_image(tmp_img)
                cv2.imwrite(
                    f"./data/{chess_pieces[i]}/{chess_pieces[i]}_{cnt}.png", tmp_img
                )
                cnt += 1

            # cv2.imwrite(f"./data/{chess_pieces[i]}/{chess_pieces[i]}_{iter}.png", roi)

        # cv2.imwrite(f"./data/chess_piece_{i}.png", roi)

        # print(f"Saving {chess_pieces[i]}.png")
        line_progress.update(i * 100 // len(circles[0, :]))
        # check_chess_piece(roi, i)


# Display the original image with detected circles
for i in circles[0, :]:
    cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), -1)

# Show the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
