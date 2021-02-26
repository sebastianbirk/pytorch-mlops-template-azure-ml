import copy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from azureml.core.run import Run
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


def eval_model(model, inputs):
    return {"acc": 1}
    # outputs = model(inputs)
    # _, preds = torch.max(outputs, 1)
    # running_correct_preds += torch.sum(preds == labels.data)
    # epoch_acc = running_correct_preds.double() / dataset_sizes[phase]


def train_model(
    model: torchvision.models,
    criterion: torch.nn.modules.loss,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    num_epochs: int,
    dataloaders: dict,
    dataset_sizes: dict
 ) -> torchvision.models:
    """
    Train the model and track training and validation loss and accuracy.
    :param model: pretrained model which will be trained further
    :param criterion: torch loss function
    :param optimizer: torch optimizer
    :param scheduler: torch learning rate scheduler
    :param num_epochs: number of epochs to train the model
    :param dataloaders: dictionary of torch dataloaders
    :param dataset_sizes: dictionary with lengths of the training and val set
    :return model: pretrained model with tuned final fully connected layer
    """
    
    # Get the run context
    run = Run.get_context()

    # Leverage GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start_time = time.time()

    # Load in weights of model
    best_model_weights = copy.deepcopy(model.state_dict())
    
    # Initialize best_acc
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("=" * 20)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("=" * 20)

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

                    # Backward pass, gradient optimization and learning rate update
                    # only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        scheduler.step() 

                # Calculate statistics
                running_loss += loss.item() * inputs.size(0)
                running_correct_preds += torch.sum(preds == labels.data)
                

            # Average loss and accuracy over examples
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct_preds.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            # Log the epoch validation loss and accuracy to AML run
            if phase == "train":
                run.log("train_loss", np.float(epoch_loss))
                run.log("train_acc", np.float(epoch_acc))
                
            if phase == "val":
                run.log("val_loss", np.float(epoch_loss))
                run.log("val_acc", np.float(epoch_acc))

            # Deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

                # Log the best val accuracy to AML run
                run.log("best_val_acc", np.float(best_acc))
            
            if phase == "train":
                print("-" * 20)

    time_elapsed = time.time() - start_time
    
    print(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.")
    print(f"Best Val Acc: {best_acc:4f}")
          
    # Load best model weights
    model.load_state_dict(best_model_weights)
          
    return model


def fine_tune_model(
    num_epochs: int,
    num_classes: int,
    dataloaders: dict,
    dataset_sizes: dict,
    learning_rate: float,
    momentum: float
) -> torchvision.models:
    """
    Load a pretrained model and reset the final fully connected layer.
    :param num_epochs: number of epochs to train the model
    :param num_classes: number of target classes 
        (supports binary and multiclass classification)
    :param dataloaders: dictionary of torch dataloaders
    :param dataset_sizes: dictionary with lengths of the training and val set
    :param learning_rate: learning rate hyperparameter
    :param momentum: momentum hyperparameter
    :return model: pretrained model with tuned final fully connected layer
    """

    print("=" * 20)
    print("START TRAINING")
    print("=" * 20)
    
    # Get the run context
    run = Run.get_context()

    # Log the hyperparameter metrics to the AML run
    run.log("lr", learning_rate)
    run.log("momentum", momentum)

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