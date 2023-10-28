import os
import PIL
import numpy as np
import torch

class dataset_folder(torch.utils.data.Dataset):
    def __init__(self, imgs_path, transform=None, augmentations=None):
        self.data_paths = []
        self.label_paths = []
        self.augmentations = augmentations
        self.transform = transform

        for folder in os.listdir(imgs_path):
            class_name = folder
            folder_path = os.path.join(imgs_path, folder)
            for img_name in sorted(os.listdir(folder_path)):
                full_path = os.path.join(folder_path, img_name)
                self.data_paths.append([full_path, class_name])

        self.classes = {"drinking" : 0, "eating": 1,
                           'interacting_with_phone': 2, 'sitting_still': 3,
                             'talking_on_phone': 4}
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img, class_name = self.data_paths[idx]
        img = PIL.Image.open(img)
        if self.img_augmentations is not None:
            img = np.array(img)
            img = self.augmentations(image=img)['image']
            img = PIL.Image.fromarray(img)
            
        img = self.transform(img)
        class_id = self.classes[class_name]
        class_id = torch.tensor(class_id)
        
        return img, class_id

class data_loader:
    def __init__(self, dataset, split=0.8, batch_size=16, num_workers=4):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def loader(self):
        train_size = int(self.split * len(self.dataset))
        val_size = len(self.dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=self.num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=self.num_workers)

        return train_loader, val_loader


