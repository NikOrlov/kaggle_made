import os
import re
import pickle
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import config


class SportDataset(Dataset):
    def __init__(self, root='Dataset', stage='train', transform=None):
        self.images = os.listdir(os.path.join(root, stage))
        self.csv = pd.read_csv(os.path.join(root, f'{stage}.csv'))
        if stage == 'test':
            self.csv['label'] = ['football'] * len(self.csv)
        self.labels = self.csv.set_index('image_id').to_dict()['label']
        self.names = list(self.labels.keys())
        with open(config.ADDITIONAL, 'rb') as file:
            self.classes, self.label_to_id, self.id_to_label = pickle.load(file)

        self.stage = stage

        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        file = self.names[index]
        label = self.labels[file]
        label = self.label_to_id[label]

        img = np.array(Image.open(os.path.join(self.root, self.stage, file)))

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img, label
