import torch
import numpy as np
from matplotlib import pyplot as plt
import random
import os
from torch.utils.data import Dataset, DataLoader
from models.seven_conv import seven_conv
from models.two_conv import two_conv
from numpy import linalg as LA

class SimulatedDataset(Dataset):
    def __init__(self):
        self.len_data = 2
        self.work_dir = './data/task3/images'
        self.data = self.create_images_and_labels(len_data=50)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        frame, label = self.data[idx]
        sample = (frame, label)

        return sample

    def __len__(self):
        return len(self.data)

    def create_image_and_label(self):
        frame = np.zeros([28, 28])
        tl_coord = [random.randrange(-10, 10) for j in range(2)]
        tl_x, tl_y = tl_coord[0], tl_coord[1]
        frame[tl_x:, :] = 1
        frame[:, tl_y:] = 1
        tl_coord = np.array(tl_coord)
        frame = np.array(frame)
        frame = np.expand_dims(frame, axis=0)

        return [frame, tl_coord]

    def create_images_and_labels(self, len_data):
        data = []
        for idx in range(len_data):
            instance = self.create_image_and_label()
            data.append(instance)
        return data


set = SimulatedDataset()
set_loader = DataLoader(set, batch_size=4,
                        shuffle=True, num_workers=0)


seven_conv_model = seven_conv()
two_conv_model = two_conv()


model = two_conv_model
model = model.double()


# optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
#                                    model.parameters()),
#                             lr=0.005,
#                             momentum=0.9)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()),
                             lr=0.005)

criterion = torch.nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(
          model,
          set_loader,
          criterion,
          optimizer,
          device):

    losses=[]

    epoch_n = 50
    model = model.train()

    for epoch in range(1, epoch_n + 1):
        for batch_id, (image, label) in enumerate(set_loader):
            label, image = label.to(device), image.to(device)
            output = model(image)
            loss = criterion(output, label.double())
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def compute_accuracy(model, data_set, device):
    model = model.eval()
    with torch.no_grad():
        mse = 0
        total = 0
        for batch_id, (image, label) in enumerate(data_set):
            image = image.to(device)
            label = label.to(device)
            outputs = model(image).to(device)
            se = LA.norm(outputs-label.double())
            print(se)
            total += label.shape[0]/4*2
            mse += se
    print(mse)
    return mse

train(model=model, set_loader=set_loader, criterion=criterion, optimizer=optimizer, device=device)
compute_accuracy(model, set_loader, device)