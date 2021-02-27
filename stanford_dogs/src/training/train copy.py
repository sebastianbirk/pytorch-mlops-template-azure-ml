# Append root folder to sys path to be able to import src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

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
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from zipfile import ZipFile

from utils import load_data

run = Run.get_context()


def train_model(model: torchvision.models,
                criterion: torch.nn.modules.loss,
                optimizer: torch.optim,
                scheduler: torch.optim.lr_scheduler,
                num_epochs: int,
                dataloaders: dict,
                dataset_sizes: dict) -> torchvision.models:
    """
    Train the model on the stanford dogs dataset and track training
    and validation loss and accuracy.
    :param model: pretrained model which will be trained further
    :param criterion: torch loss function
    :param optimizer: torch optimizer
    :param scheduler: torch learning rate scheduler
    :param num_epochs: number of epochs to train the model
    :param dataloaders: dictionary of torch dataloaders
    :param dataset_sizes: dictionary with lengths of the training, val and test set
    :return model: pretrained model with tuned final fully connected layer
    """
    
    # Leverage GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start_time = time.time()

    # Load in weights of model
    best_model_weights = copy.deepcopy(model.state_dict())
    
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("-" * 20)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode

            running_loss = 0.0
            running_correct_preds = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only if in training phase
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and gradient optimization only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Calculate statistics
                running_loss += loss.item() * inputs.size(0)
                running_correct_preds += torch.sum(preds == labels.data)
                
            # Update learning rate if in training phase
            if phase == "train":
                scheduler.step() 

            # Average loss and accuracy over examples
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct_preds.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

            # Log the best val accuracy to AML run
            run.log("best_val_acc", np.float(best_acc))
            print("-" * 20)

    time_elapsed = time.time() - start_time
    
    print(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.")
    print(f"Best Val Acc: {best_acc:4f}")
          
    # Load best model weights
    model.load_state_dict(best_model_weights)
          
    return model


def fine_tune_model(num_epochs: int,
                    num_classes: int,
                    dataloaders: dict,
                    dataset_sizes: dict,
                    learning_rate: float,
                    momentum: float) -> torchvision.models:
    """
    Load a pretrained model and reset the final fully connected layer.
    :param num_epochs: number of epochs to train the model
    :param num_classes: number of target classes 
        (supports binary and multiclass classification)
    :param dataloaders: dictionary of torch dataloaders
    :param dataset_sizes: dictionary with lengths of the training, val and test set
    :param learning_rate: learning rate hyperparameter
    :param momentum: momentum hyperparameter
    :return model: pretrained model with tuned final fully connected layer
    """

    print("-" * 20)
    print("START TRAINING")
    print("-" * 20)
    
    # Log the hyperparameter metrics to the AML run
    run.log("lr", np.float(learning_rate))
    run.log("momentum", np.float(momentum))

    # Load pretrained model and reset final fully connected layer to have num_classes output neurons
    model_ft = models.resnet18(pretrained=True) 
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    # Leverage GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    # Specify loss function
    criterion = nn.CrossEntropyLoss()

    # Create SGD optimizer to optimize all parameters
    optimizer_ft = optim.SGD(model_ft.parameters(),
                             lr=learning_rate,
                             momentum=momentum)
                            
    # Create scheduler to decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                           step_size=7,
                                           gamma=0.1)
    
    # Start model training
    model = train_model(model_ft, criterion, optimizer_ft,
                        exp_lr_scheduler, num_epochs, dataloaders,
                        dataset_sizes)

    return model


def main():
    
    print("Torch version:", torch.__version__)
    
    # Retrieve command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path where the images are stored")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs to train")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    args = parser.parse_args()
         
    print("-" * 20)
    print("LOAD DATA")      
    print("-" * 20)
          
    # Load training and validation data
    dataloaders, dataset_sizes, class_names = load_data(args.data_path)
          
    print("Data has been load successfully.")
        
    # Train the model
    model = fine_tune_model(num_epochs=args.num_epochs,
                            num_classes=len(class_names),
                            dataloaders=dataloaders,
                            dataset_sizes=dataset_sizes,
                            learning_rate=args.learning_rate,
                            momentum=args.momentum)
    
    # Save the model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model, os.path.join(args.output_dir, "model.pt"))
    print("-" * 20)
    print(f"Model saved in {args.output_dir}.")


if __name__ == "__main__":
    main()
