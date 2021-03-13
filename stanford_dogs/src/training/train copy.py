# Append src folder to sys path to be able to import modules from src
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import libraries
import argparse
import copy
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from azureml.core import Run

# Import created modules
from utils import load_data

# Get run context
run = Run.get_context()


def train_model(model: torchvision.models,
                criterion: torch.nn.modules.loss,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                num_epochs: int,
                dataloaders: dict,
                dataset_sizes: dict) -> torchvision.models:
    """
    Train the model on the stanford dogs dataset using transfer learning and track training
    and validation loss and accuracy.
    :param model: pretrained torchvision model which will be trained further
    :param criterion: torch loss function
    :param optimizer: torch optimizer
    :param scheduler: torch learning rate scheduler
    :param num_epochs: number of epochs to train the model
    :param dataloaders: dictionary of torch dataloaders containing train and val dataloader
    :param dataset_sizes: dictionary with lengths of the train and val set
    :return model: tuned torchvision model
    """
    
    # Leverage GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Track training start time to measure duration
    start_time = time.time()

    # Load in weights of pretrained model
    # Check https://docs.python.org/3/library/copy.html for copy interface
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

            # Iterate over data batches
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history for gradient updates only if in training phase
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and gradient optimization only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Calculate loss and correct predictions
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
            run.log("best_val_acc", float(best_acc))
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
    run.log("lr", float(learning_rate))
    run.log("momentum", float(momentum))

    # Load pretrained model and reset final fully connected layer to have num_classes output neurons
    # model_ft = models.resnet18(pretrained=True) 
    model_ft = torchvision.models.resnext50_32x4d(pretrained=True)

    # Freeze all network layers except for the ones we add. 
    # With this, we basically use the ConvNet layers as fixed feature extractor.
    # We need to set requires_grad == False to freeze the parameters
    # so that the gradients are not computed in backward()
    # for param in model_ft.parameters():
    #     print(param)
    #     param.requires_grad = False
    for i, child in enumerate(model_ft.children()):
        # print(i)
        # print(child)
        if i <= 7:
            for param in child.parameters():
                param.requires_grad = False

    # Modify last layer and add additional linear layer.
    # This is basically our classifier that we will train on top of fixed feature extractor
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_ft.fc.in_features # this is 512/2048
    # model_ft.fc = nn.Sequential(nn.Linear(num_ftrs , 256),
    #                             nn.BatchNorm1d(256),
    #                             nn.Dropout(0.2),
    #                             nn.Linear(256 , num_classes))
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs , 512),
                                nn.BatchNorm1d(512),
                                #nn.Dropout(0.2),
                                #nn.Linear(1024 , 512),
                                #nn.BatchNorm1d(512),
                                # nn.Dropout(0.2),
                                #nn.Linear(512 , 256),
                                #nn.BatchNorm1d(256),
                                # nn.Dropout(0.2),
                                nn.Linear(512 , num_classes))

    # Leverage GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    # Specify loss function
    criterion = nn.CrossEntropyLoss()

    # Create SGD optimizer to optimize all parameters
    # optimizer_ft = optim.SGD(model_ft.parameters(),
    #                          lr=learning_rate,
    #                          momentum=momentum)

    # Create ADAM optimizer to optimize all parameters
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()),
                              lr=learning_rate,
                              betas=(momentum, 0.999))
                            
    # Create scheduler to decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft,
                                                 step_size=10,
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
    torch.save(model, os.path.join(args.output_dir, "dog_clf_model.pt"))
    print("-" * 20)
    print(f"Model saved in {args.output_dir}.")


if __name__ == "__main__":
    main()
