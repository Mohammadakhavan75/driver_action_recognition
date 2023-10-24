# eval
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import cv2
import PIL
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.blur import transforms as a_t
import glob
from tqdm import tqdm
import shutil
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_path, transform=None, transform2=None):
        self.data_paths = []
        self.label_paths = []
        self.transform = transform
        self.transform2 = transform2

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
        if self.transform is not None:
            img = np.array(img)
            img = self.transform(image=img)['image']
            img = PIL.Image.fromarray(img)
            
        img = self.transform2(img)
        class_id = self.classes[class_name]
        class_id = torch.tensor(class_id)
        
        return img, class_id


def preprocessing(data_dir, batch_size=32, num_workers=8, image_size=224):
    data_transforms = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = MyDataset(os.path.join(data_dir),
                                        None,
                                        data_transforms)

    val_loader = torch.utils.data.DataLoader(full_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)

    class_names = full_dataset.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return val_loader, class_names, device


def load_model(device,
               model_path=None):
        
    model = models.mobilenet_v2(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 784)
    model.classifier.append(nn.Dropout(p=0.2))
    model.classifier.append(nn.Linear(784,512))
    model.classifier.append(nn.Dropout(p=0.2))
    model.classifier.append(nn.Linear(512,128))
    model.classifier.append(nn.Dropout(p=0.2))
    model.classifier.append(nn.Linear(128,5))
    state_dict = torch.load(model_path)
    # load the state dict into the model
    model.load_state_dict(state_dict)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    return model, criterion


def eval(val_loader,
        model,
        criterion,
        device):
    
    since = time.time()
    phase='val'
    model.eval()
    eval_losses = []
    eval_acc = []
    for data_ in tqdm(val_loader):
        inputs, labels = data_
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            acc = (torch.sum(preds == labels.data).detach().cpu().numpy())/len(preds)
            eval_losses.append(loss.item())
            eval_acc.append(acc)

    time_elapsed = time.time() - since
    print(f'Evaluation complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Evaluation Acc: {np.mean(eval_acc):.4f} Loss: {np.mean(eval_losses):.4f}')


if __name__ == '__main__':
    batch_size=16
    image_size=224
    num_workers=16
    data_dir = "/SSD/DriverActivity/DriverActivityRecognition/Drive&Act/SelectedData/"
    model_path = "/home/makhavan/action_recognition/re_train/run/exp-2023-10-17-16-13-22-342059/models/best_params.pt"
    
    print("Preprocessing data...")
    val_loader, class_names, device = preprocessing(data_dir, batch_size=batch_size, image_size=image_size, num_workers=num_workers)
    print("Loading pre-train model...")
    model, criterion = load_model(device, model_path=model_path)
    print("Start evaluating...")
    eval(val_loader, model, criterion, device)
