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


def train_model(dataloaders: dict,
                dataset_sizes: dict,
                num_classes: int,
                num_epochs: int,
                batch_size: int,
                learning_rate: float,
                momentum: float,
                num_frozen_layers: int,
                num_neurons_fc_layer: int,
                dropout_prob_fc_layer: int,
                lr_scheduler_step_size: int) -> torchvision.models:
    """
    Apply transfer learning by loading a pretrained ResNext-50 model, modifying its
    network architecture and retraining it on a new dataset
    :param dataloaders: dictionary of torch dataloaders containing train and val dataloader
    :param dataset_sizes: dictionary with lengths of the train and val dataset
    :param num_classes: number of target classes 
        (supports binary and multiclass classification)
    :param num_epochs: number of epochs to train the model
    :param batch_size: batch size hyperparameter
    :param learning_rate: learning rate hyperparameter
    :param momentum: momentum hyperparameter
    :param num_frozen_layers: num_frozen_layers hyperparameter
        (indicates how many layers of the pretrained model will be excluded from param optimization)
    :param num_neurons_fc_layer: num_neurons_fc_layer hyperparamter
        (indicates how many neurons the last fully connected layer has)
    :param dropout_prob_fc_layer: dropout_prob_fc_layer hyperparameter
        (indicates the dropout probability for the last fully connected layer)
    :param lr_scheduler_step_size: lr_scheduler_step_size hyperparameter
        (indicates after how many epochs the learning rate will be reduced)
    :return model: tuned ResNext-50 model
    """

    print("-" * 20)
    print("START MODEL TRAINING")
    print("-" * 20)
    
    print(f"Hyperparameter number of epochs: {int(num_epochs)}")
    print(f"Hyperparameter batch size: {int(batch_size)}")
    print(f"Hyperparameter learning rate: {float(learning_rate)}")
    print(f"Hyperparameter momentum: {float(momentum)}")
    print(f"Hyperparameter number of frozen layers: {int(num_frozen_layers)}")
    print(f"Hyperparameter number of neurons fc layer: {int(num_neurons_fc_layer)}")
    print(f"Hyperparameter dropout probability fc layer: {int(dropout_prob_fc_layer)}")
    print(f"Hyperparameter lr scheduler step size: {int(lr_scheduler_step_size)}")

    # Log the hyperparameter metrics to the AML run
    run.log("num_epochs", int(num_epochs))
    run.log("batch_size", int(batch_size))
    run.log("lr", float(learning_rate))
    run.log("momentum", float(momentum))
    run.log("num_frozen_layers", int(num_frozen_layers))
    run.log("num_neurons_fc_layer", int(num_neurons_fc_layer))
    run.log("dropout_prob_fc_layer", int(dropout_prob_fc_layer))
    run.log("lr_scheduler_step_size", int(lr_scheduler_step_size))

    # Load pretrained ResNext-50 model
    model_ft = torchvision.models.resnext50_32x4d(pretrained=True)

    # Freeze <num_frozen_layers> network layers of the pretrained model. 
    # With this, we basically have control over how many (ConvNet) layers to use 
    # as fixed feature extractor.
    # We need to set requires_grad = False to freeze the parameters
    # so that the gradients are not computed in backward()
    for i, child in enumerate(model_ft.children()):
        if i <= num_frozen_layers:
            for param in child.parameters():
                param.requires_grad = False

    # Modify last layer and add additional linear layer.
    # This is basically our classifier that we will train on top of the feature extractor
    # Parameters of newly constructed modules have requires_grad = True by default
    num_ftrs = model_ft.fc.in_features # this is the number of input neurons for the classifier 
                                       # which is 2048 for the ResNext-50
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs , num_neurons_fc_layer),
                                nn.BatchNorm1d(num_neurons_fc_layer),
                                nn.Dropout(dropout_prob_fc_layer),
                                nn.Linear(num_neurons_fc_layer, num_classes))

    # Leverage GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    # Specify loss function
    criterion = nn.CrossEntropyLoss()

    # Create ADAM optimizer to optimize all parameters
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()),
                              lr=learning_rate,
                              betas=(momentum, 0.999))
                            
    # Create scheduler to decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft,
                                                 step_size=lr_scheduler_step_size,
                                                 gamma=0.1)
    
    # Start model tuning
    model = fine_tune_model(model_ft, criterion, optimizer_ft,
                            exp_lr_scheduler, num_epochs, dataloaders,
                            dataset_sizes)

    return model


def fine_tune_model(model: torchvision.models,
                    criterion: torch.nn.modules.loss,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler,
                    num_epochs: int,
                    dataloaders: dict,
                    dataset_sizes: dict) -> torchvision.models:
    """
    Train the pretrained model on the given dataset and track training
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

    # Initialize model weights with weights of pretrained model
    # Check https://docs.python.org/3/library/copy.html for copy interface
    best_model_weights = copy.deepcopy(model.state_dict())
    
    best_acc = 0.0
 
    # Iterate over training epochs
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
            cross_entropy_running = 0

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
                    
                    cross_entropy_current_batch = nn.functional.cross_entropy(outputs, labels)
                    cross_entropy_running += cross_entropy_current_batch
                    # print(f"Preds: {preds}")
                    # print(f"Labels: {labels}")
                    # print(f"Cross Entropy: {cross_entropy_current_batch}")

                    # Backward pass and gradient optimization only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Calculate loss and correct predictions
                running_loss += loss.item() * inputs.size(0) # Averaged loss times batch size
                running_correct_preds += torch.sum(preds == labels.data) 

            # Average loss and accuracy over examples
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct_preds.double() / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} {phase.capitalize()} Acc: {epoch_acc:.4f}")
            print(f"Cross Entropy: {cross_entropy_running}")
            
            if phase == "train":
                # Track train loss and acc
                run.log("train_loss", float(epoch_loss))
                run.log("train_acc", float(epoch_acc))

                # Update learning rate if in training phase
                scheduler.step()

            elif phase == "val":
                # Track val loss and acc
                run.log("val_loss", float(epoch_loss))
                run.log("val_acc", float(epoch_acc))

                # Save the new model weights if the val epoch_acc is > best_acc at that point in time
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())

        # Log the best val accuracy to AML run
        run.log("best_val_acc", float(best_acc))
        print("-" * 20)

    # Calculate training duration
    time_elapsed = time.time() - start_time

    # Log the training duration to AML run
    run.log("training_duration_min", time_elapsed // 60)
    
    print(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best validation accuracy: {best_acc:4f}")
          
    # Load best model weights into model
    model.load_state_dict(best_model_weights)
          
    return model


def main():
    
    print("Torch version:", torch.__version__)

    # Retrieve command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path where the images are stored")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs to train")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--num_frozen_layers", type=int, default=7, help="Number of frozen layers")
    parser.add_argument("--num_neurons_fc_layer", type=int, default=512, help="Number of neurons of the fc layer")
    parser.add_argument("--dropout_prob_fc_layer", type=float, default=0.0, help="Dropout probability of the fc layer")
    parser.add_argument("--lr_scheduler_step_size", type=int, default=10, help="Step size for lr scheduler (epochs)")
    args = parser.parse_args()
         
    print("-" * 20)
    print("CREATE DATALOADERS")      
    print("-" * 20)
          
    # Load training and validation data
    dataloaders, dataset_sizes, class_names = load_data(args.data_path,
                                                        args.batch_size)
          
    print("Dataloaders have been created successfully.")
        
    # Train the model
    model = train_model(dataloaders=dataloaders,
                        dataset_sizes=dataset_sizes,
                        num_classes=len(class_names),
                        num_epochs=args.num_epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        momentum=args.momentum,
                        num_frozen_layers=args.num_frozen_layers,
                        num_neurons_fc_layer=args.num_neurons_fc_layer,
                        dropout_prob_fc_layer=args.dropout_prob_fc_layer,
                        lr_scheduler_step_size=args.lr_scheduler_step_size)
    
    # Save the model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model, os.path.join(args.output_dir, "dog_clf_model.pt"))
    print("-" * 20)
    print(f"Model saved in {args.output_dir}.")


if __name__ == "__main__":
    main()
