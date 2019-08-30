from torch.utils.data import Dataset
import numpy as np
import random


class SimulatedDataset(Dataset):
    def __init__(self, len_data, dim=28):
        self.len_data = len_data
        self.dim = dim
        self.data = self.create_images_and_labels(len_data)

    def __getitem__(self, idx):
        frame, label = self.data[idx]
        sample = (frame, label)
        return sample

    def __len__(self):
        return len(self.data)

    def create_image_and_label(self):
        frame = np.zeros([self.dim, self.dim])
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
