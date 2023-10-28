# train v0.3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from get_models import get_models
import argparse
from preprocessing import preprocessing
from data_loader import dataset_folder, data_loader
def parsing():
    parser = argparse.ArgumentParser(description='',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img_path', help='Path for image file', type=str, default=None)
    parser.add_argument('--vid_path', help='Path for video file', type=str, default=None)
    parser.add_argument('--folder_path', help='Path for video file', type=str, default=None)
    parser.add_argument('--model_path', help='Model path file', type=str, required=True)
    parser.add_argument('--device', help='Device can be cuda or cpu or None', type=str, default=None)
    parser.add_argument('--gpu_num', help='GPU number', type=int, default=0)
    args = parser.parse_args()
    args.armnn_delegate = None

    return args



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
        if "vit" in model_name:
            for param in model.encoder.parameters():
                param.requires_grad = False
        else:
            for param in model.features.parameters():
                param.requires_grad = False

    if fine_tune:
        if "vit" in model_name:
            params = list(model.features.parameters())
            for param in params[:-fine_tune_layers]:
                param.requires_grad = False
        else:
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
    model_name = "vit_b_16"
    model_layers = [5]
    model_path = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    addr = datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')
    save_path = './run/exp-' + addr + '/'
    model_save_path = save_path + 'models/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)

    writer = SummaryWriter(save_path)

    args = parsing()
    print("Preprocessing data...")
    preproc = preprocessing(image_size=image_size, num_transfroms=3)
    img_transforms_torch, img_augmentations = preproc.loading_transforms()
    
    if args.folder_path is not None:
        dataset = dataset_folder(imgs_path=data_dir, transform=img_transforms_torch, augmentations=img_augmentations)

    loader = data_loader(dataset=dataset, split=0.8, batch_size=batch_size, num_workers=num_workers)
    train_loader, val_loader = loader.loader()

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
