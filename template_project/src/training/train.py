import argparse
import copy
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import urllib

from azureml.core import Run
from data_utils import load_data
from model import Net
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from zipfile import ZipFile

run = Run.get_context()


def train_model(model, criterion, optimizer, scheduler, num_epochs, data_dir):
    """Train the model."""

    # load training/validation data
    dataloaders, dataset_sizes, class_names = load_data(data_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # log the best val accuracy to AML run
            run.log('best_val_acc', np.float(best_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def fine_tune_model(num_epochs, data_dir, learning_rate, momentum):
    """Load a pretrained model and reset the final fully connected layer."""

    # log the hyperparameter metrics to the AML run
    run.log('lr', np.float(learning_rate))
    run.log('momentum', np.float(momentum))

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)  # only 2 classes to predict

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(),
                             lr=learning_rate, momentum=momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model_ft, criterion, optimizer_ft,
                        exp_lr_scheduler, num_epochs, data_dir)

    return model


def main():
    print("Torch version:", torch.__version__)

    # Retrieve command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the data")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs to train")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    
    args = parser.parse_args()
    
    model = fine_tune_model(args.num_epochs, args.data_path,
                            args.learning_rate, args.momentum)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    torch.save(model, os.path.join(args.output_dir, "model.pt"))


if __name__ == "__main__":
    main()
