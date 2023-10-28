# eval
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import time
import os
import PIL
from tqdm import tqdm
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
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
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
    # device = 'cpu'

    return val_loader, class_names, device


def load_model(model_name,
               layers,
               model_path,
               device
               ):
        
    get_model = get_models(model_name, layers)
    model = get_model.get_model()

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    
    return model, criterion


def eval(val_loader,
        model,
        criterion,
        device):
    
    since = time.time()
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
    model_path = "/home/makhavan/driver_action_recognition/run/exp-2023-10-24-18-03-41-530238/models/best_params.pt"
    model_name = "vit_b_16"
    model_layers = [5]
    
    print("Preprocessing data...")
    val_loader, class_names, device = preprocessing(data_dir, batch_size=batch_size, image_size=image_size, num_workers=num_workers)
    print("Loading pre-train model...")
    model, criterion = load_model(model_name, model_layers, model_path, device)
    print("Start evaluating...")
    eval(val_loader, model, criterion, device)
