import glob
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import global_params

import torch
from torch import nn

from torch.nn import (
    Conv2d,
    Linear,
    ReLU,
    MaxPool2d,
    Flatten,
    Sequential,
    CrossEntropyLoss,
)

from torch.utils.tensorboard.writer import SummaryWriter


# 通过创建data.Dataset子类Mydataset来创建输入
class Mydataset(data.Dataset):
    # 类初始化
    def __init__(self, root):
        self.imgs_path = root

    # 进行切片
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        return img_path

    # 返回长度
    def __len__(self):
        return len(self.imgs_path)


# glob: get all paths of images
all_img_paths = glob.glob(r"./data/*/*.png")
# CHECK: loop to show paths of all images
# for var in all_img_paths:
#     print(var)


# Use the custom class Mydataset to create an object weather_dataset
dataset = Mydataset(all_img_paths)
print(len(dataset))


chess_piece_types = global_params.Chess_pieces_types
species_to_id = dict((c, i) for i, c in enumerate(chess_piece_types))
print(species_to_id)
id_to_species = dict((v, k) for k, v in species_to_id.items())
print(id_to_species)
all_labels = []

for img in all_img_paths:
    # distinguish what category each img should belong to
    for i, c in enumerate(chess_piece_types):
        if c in img:
            all_labels.append(i)
# print(all_labels)

size = global_params.Size

# transform the data to tensor
transform = transforms.Compose(
    [
        transforms.Resize((size, size)),  # 第一步转换
        transforms.ToTensor(),  # 第二步转换，作用：第一转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
    ]
)


class Mydatasetpro(data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    # slice the data
    def __getitem__(
        self, index
    ):  # 根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img)
        data = self.transforms(pil_img)
        return data, label

    # return the length
    def __len__(self):
        return len(self.imgs)


def show_imgs(imgs_batch, labels_batch):
    plt.figure(figsize=(12, 8))
    for i, (img, label) in enumerate(zip(imgs_batch[:6], labels_batch[:6])):
        img = img.permute(1, 2, 0).numpy()
        plt.subplot(2, 3, i + 1)
        plt.title(id_to_species.get(label.item()))
        plt.imshow(img)
    plt.show()


# shuffle the data and partition the data into train and test
index = np.random.permutation(len(all_img_paths))

all_img_paths = np.array(all_img_paths)[index]
all_labels = np.array(all_labels)[index]


# TODO: revise the batch size
BATCH_SIZE = 50

dataset = Mydatasetpro(all_img_paths, all_labels, transform)
datalodaer = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# CHECK: show the images
# imgs_batch, labels_batch = next(iter(datalodaer))
# show_imgs(imgs_batch, labels_batch)

# 80% as train
# TODO: revise the train_dataset_rate
train_dataset_rate = 0.7
s = int(len(all_img_paths) * train_dataset_rate)
print(s)

train_imgs = all_img_paths[:s]
train_labels = all_labels[:s]
train_dataset = Mydatasetpro(train_imgs, train_labels, transform)  # TrainSet TensorData
train_dataloader = data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)  # TrainSet Labels

test_imgs = all_img_paths[s:]
# test_labels = all_imgs_path[s:] # RECORD: This is a fucking trivial error, which waste a lot of my time
test_labels = all_labels[s:]
test_dataset = Mydatasetpro(test_imgs, test_labels, transform)  # TestSet TensorData
test_dataloader = data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)  # TestSet Labels


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            ReLU(),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            ReLU(),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(64 * 16 * 16, 64 * 4),
            Linear(64 * 4, 14),
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = Model()
# model = torch.load("models/model.pth")

loss_fn = CrossEntropyLoss()

# TODO: revise learning rate
learning_rate = 0.01

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
)

epochs = 100
train_step = 0

writer = SummaryWriter("logs/test")

for epoch in range(epochs):
    print(f"------ The {epoch}-th train begin ------")

    # Step 1: Train the model
    model.train()  # like eval(), only affects specific layers BatchNorm and Dropout, so this model can be omitted
    for data in train_dataloader:
        images, labels = data

        labels = torch.tensor(labels, dtype=torch.long)
        outputs = model(images)

        # check the property
        # print(images, labels, outputs)
        # print(images.shape, labels.shape, outputs.shape)
        # print(images.dtype, outputs.dtype, labels.dtype)

        loss = loss_fn(outputs, labels)

        # optimizer optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step += 1
        if train_step % 100 == 0:
            print(f"the {train_step}-th train step, loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), train_step)

    # Step 2: Test the model
    correct = 0
    model.eval()  # only affects specific layers BatchNorm and Dropout, so this model can be omitted
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            labels = torch.tensor(labels, dtype=torch.long)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # print(outputs.argmax(1), labels)
            correct += (outputs.argmax(1) == labels).sum()

    accuracy = correct / len(test_dataset)
    print(f"epoch: {epoch}, test_loss: {loss}, test_accuracy: {accuracy*100}%")
    writer.add_scalar("test_loss", loss, epoch)
    writer.add_scalar("test_accuracy", accuracy, epoch)

    if accuracy > 0.96:
        torch.save(model, f"models/model.pth")
        exit()
        break

    if epoch % 10 == 0:
        torch.save(model, f"models/model_{int(epoch/10)}.pth")
        print(f"model_{int(epoch/10)}.pth saved")


writer.close()
