import glob
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import Global_Params

import torch
from torch import nn
from torch.nn import functional as F

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

import torchvision
from torch.utils.data import DataLoader


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


# 使用glob方法来获取数据图片的所有路径
all_imgs_path = glob.glob(r"./data/*/*.png")  # 数据文件夹路径，根据实际情况更改！
# 循环遍历输出列表中的每个元素，显示出每个图片的路径
# for var in all_imgs_path:
# print(var)

# 利用自定义类Mydataset创建对象weather_dataset
weather_dataset = Mydataset(all_imgs_path)
print(len(weather_dataset))  # 返回文件夹中图片总个数


chess_pieces_types = Global_Params.Chess_pieces_types
species_to_id = dict((c, i) for i, c in enumerate(chess_pieces_types))
print(species_to_id)
id_to_species = dict((v, k) for k, v in species_to_id.items())
print(id_to_species)
all_labels = []
# 对所有图片路径进行迭代
for img in all_imgs_path:
    # 区分出每个img，应该属于什么类别
    for i, c in enumerate(chess_pieces_types):
        if c in img:
            all_labels.append(i)
# print(all_labels)  # 得到所有标签

size = Global_Params.Size
# 对数据进行转换处理
transform = transforms.Compose(
    [
        # transforms.Resize((96, 96)),  # 做的第一步转换
        transforms.Resize((size, size)),  # 做的第一步转换
        transforms.ToTensor(),  # 第二步转换，作用：第一转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
    ]
)


class Mydatasetpro(data.Dataset):
    # 类初始化
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    # 进行切片
    def __getitem__(
        self, index
    ):  # 根据给出的索引进行切片，并对其进行数据处理转换成Tensor，返回成Tensor
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img)
        data = self.transforms(pil_img)
        return data, label

    # 返回长度
    def __len__(self):
        return len(self.imgs)


BATCH_SIZE = 50
weather_dataset = Mydatasetpro(all_imgs_path, all_labels, transform)
wheather_datalodaer = data.DataLoader(
    weather_dataset, batch_size=BATCH_SIZE, shuffle=True
)

imgs_batch, labels_batch = next(iter(wheather_datalodaer))
print(imgs_batch.shape)

plt.figure(figsize=(12, 8))
for i, (img, label) in enumerate(zip(imgs_batch[:6], labels_batch[:6])):
    img = img.permute(1, 2, 0).numpy()
    plt.subplot(2, 3, i + 1)
    plt.title(id_to_species.get(label.item()))
    plt.imshow(img)
# plt.show()  # 展示图片


# 划分测试集和训练集
index = np.random.permutation(len(all_imgs_path))

all_imgs_path = np.array(all_imgs_path)[index]
all_labels = np.array(all_labels)[index]

# 80% as train
train_dataset_rate = 0.8
s = int(len(all_imgs_path) * train_dataset_rate)
print(s)

train_imgs = all_imgs_path[:s]
train_labels = all_labels[:s]
test_imgs = all_imgs_path[s:]
# test_labels = all_imgs_path[s:]
test_labels = all_labels[s:]
train_dataset = Mydatasetpro(train_imgs, train_labels, transform)  # TrainSet TensorData
test_dataset = Mydatasetpro(test_imgs, test_labels, transform)  # TestSet TensorData
train_dataloader = data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)  # TrainSet Labels
test_dataloader = data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)  # TestSet Labels


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


model = Model()
# model = torch.load("models/model.pth")

loss_fn = CrossEntropyLoss()

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
    model.train()  # 与eval()一样，只对特定的层BatchNorm和Dropout有影响，故本模型可以不用
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
    model.eval()
    with torch.no_grad():
        # for data in train_dataloader:
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

    if epoch % 10 == 0:
        torch.save(model, f"models/model_{int(epoch/10)}.pth")
        print(f"model_{int(epoch/10)}.pth saved")

# torch.save(model, f"models/model.pth")

writer.close()
