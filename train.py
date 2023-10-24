# train v0.3
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
from get_models import get_models

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

    Image_transform = A.Compose([
        A.Resize(image_size, image_size, p=1),
        a_t.GaussianBlur(
            blur_limit=(3, 3), sigma_limit=0, always_apply=False, p=0.5),
        A.augmentations.transforms.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            brightness_by_max=True,
            always_apply=False,
            p=0.5),
        A.augmentations.transforms.RandomGamma(
            gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
        A.augmentations.transforms.RandomShadow(shadow_roi=(0, 0.5, 1, 1),
                                                num_shadows_lower=1,
                                                num_shadows_upper=2,
                                                shadow_dimension=5,
                                                always_apply=False,
                                                p=0.5),
        A.augmentations.transforms.ColorJitter(brightness=0.1,
                                               contrast=0.1,
                                               saturation=0.2,
                                               hue=0.2,
                                               always_apply=False,
                                               p=0.5),
        A.augmentations.transforms.GaussNoise(var_limit=(10.0, 50.0),
                                              mean=0,
                                              per_channel=True,
                                              always_apply=False,
                                              p=0.5),
        A.augmentations.transforms.HueSaturationValue(hue_shift_limit=20,
                                                      sat_shift_limit=30,
                                                      val_shift_limit=20,
                                                      always_apply=False,
                                                      p=0.5),
        # ToTensorV2()
    ])

    # full_dataset = datasets.ImageFolder(os.path.join(data_dir),
    #                                     Image_transform,
    #                                     data_transforms)
    full_dataset = MyDataset(os.path.join(data_dir),
                                        Image_transform,
                                        data_transforms)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)

    class_names = full_dataset.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return train_loader, val_loader, class_names, device


def init_model(model_name,
               layers,
               device,
               freeze_features=False,
               model_path=None,
               fine_tune=False,
               fine_tune_layers=24):
    
    get_model = get_models(model_name, layers)
    model = get_model.get_model()

    if model_path is not None:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    model = model.to(device)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    if fine_tune:
        params = list(model.features.parameters())
        for param in params[:-fine_tune_layers]:
            param.requires_grad = False
        
        optimizer_ft = optim.Adam(model.parameters(), lr=0.00001)
        criterion = nn.CrossEntropyLoss()
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                            step_size=5,
                                            gamma=0.9)
    else:
        # Define the loss function
        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(model.parameters())
        # optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                            step_size=10,
                                            gamma=0.9)

    return model, criterion, optimizer_ft, exp_lr_scheduler


def train_model(train_loader,
                val_loader,
                model,
                criterion,
                optimizer,
                scheduler,
                device,
                save_path,
                num_epochs=25):
    
    since = time.time()
    # Create a temporary directory to save training checkpoints
    torch.save(model.state_dict(), os.path.join(save_path,f'model_params_epoch_init.pt'))
    best_acc = 0.0
    global_iter = 0
    global_eval_iter = 0
    for epoch in range(num_epochs):
        phase='train'
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode
        dataloader = train_loader

        # Iterate over data.
        i = 0
        epoch_losses = []
        epoch_acc = []
        for data_ in tqdm(dataloader):
            i += 1
            inputs, labels = data_
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            # statistics
            epoch_losses.append(loss.item())
            acc = (torch.sum(preds == labels.data).detach().cpu().numpy())/len(preds)
            epoch_acc.append(acc)
            writer.add_scalar("Loss/train", loss.item(), global_iter)
            writer.add_scalar("Acc/train", acc, global_iter)

            global_iter += 1
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(save_path,f'model_params_epoch_{epoch}.pt'))
        

        # epoch_loss = running_loss / len(dataloader.dataset)
        # epoch_acc = running_corrects.double() / len(dataloader.dataset)
        writer.add_scalar("AVG_Loss/train", np.mean(epoch_losses), epoch)
        writer.add_scalar("AVG_Acc/train", np.mean(epoch_acc), epoch)

        phase='val'
        model.eval()  # Set model to evaluate mode
        dataloader = val_loader
        for data_ in tqdm(dataloader):
            inputs, labels = data_
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            # forward
            # track history if only in train
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                acc = (torch.sum(preds == labels.data).detach().cpu().numpy())/len(preds)
                writer.add_scalar("Loss/eval", loss.item(), global_eval_iter)
                writer.add_scalar("Acc/eval", acc, global_eval_iter)
                global_eval_iter += 1
                # backward + optimize only if in training phase
                #loss.backward()
            # statistics
            #running_loss += loss.item() * inputs.size(0)
            #running_corrects += torch.sum(preds == labels.data)

        torch.save(model.state_dict(), os.path.join(save_path,f'model_params_epoch_{epoch}.pt'))

        print(f'{phase} Loss: {np.mean(epoch_losses):.4f} Acc: {np.mean(epoch_acc):.4f}')
        # deep copy the model
        if np.mean(epoch_acc) > best_acc:
            best_acc = np.mean(epoch_acc)
            torch.save(model.state_dict(), os.path.join(save_path,'best_params.pt'))

    time_elapsed = time.time() - since
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    )
    print(f'Best val Acc: {best_acc:4f}')
    # load best model weights
    # model.load_state_dict(torch.load(best_model_params_path))

    return model


if __name__ == '__main__':
    num_epochs=50
    batch_size=16
    image_size=224
    num_workers=16
    freeze_features = True
    fine_tune= False
    fine_tune_layers = 24
    data_dir = "/SSD/DriverActivity/DriverActivityRecognition/Drive&Act/SelectedData/"
    model_name = "mobilenet_v3_small"
    model_layers = [5]
    model_path = None
    
    addr = datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')
    save_path = './run/exp-' + addr + '/'
    model_save_path = save_path + 'models/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)

    writer = SummaryWriter(save_path)
    print("Preprocessing data...")
    train_loader, val_loader, class_names, device = preprocessing(data_dir, batch_size=batch_size, image_size=image_size, num_workers=num_workers)
    print("Initializing model...")
    model, criterion, optimizer, scheduler = init_model(model_name, model_layers, device, model_path=model_path, freeze_features=freeze_features, fine_tune=fine_tune, fine_tune_layers=fine_tune_layers)
    print("Start training...")
    train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, device, model_save_path, num_epochs)


    fine_tune= True
    freeze_features = False
    fine_tune_layers = 24
    num_epochs=50
    model_path = model_save_path + 'best_params.pt'
    model_save_path = save_path + 'finetune_models/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)

    model, criterion, optimizer, scheduler = init_model(model_name, model_layers, device, model_path=model_path, freeze_features=freeze_features, fine_tune=fine_tune, fine_tune_layers=fine_tune_layers)
    print("Start finetunning...")
    train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, device, model_save_path, num_epochs)
