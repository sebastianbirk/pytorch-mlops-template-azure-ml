import os
import argparse
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from azureml.core import Run
from model import Net


run = Run.get_context()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the training data")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for SGD")
    parser.add_argument("--momentum", type=float, default=0.9,help="Momentum for SGD")

    args = parser.parse_args()

    print("")
    print("========== DATA ==========")
    print("Data Location: " + args.data_path)
    print("Available Files:", os.listdir(args.data_path))
    print("==========================")
    print("")

    # Create dataloader for CIFAR-10 training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, 
                                            download=False, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    # Leverage GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define convolutional network
    net = Net()
    net.to(device)

    # Set up pytorch cross entropy loss and SGD optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)

    print("===== MODEL TRAINING =====")
    
    # Train the network
    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Unpack the data
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad() # zero the parameter gradients

            # Run forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                loss = running_loss / 2000
                run.log("loss", loss) # log loss metric to AML
                print(f"epoch={epoch + 1}, batch={i + 1:5}: loss {loss:.2f}")
                running_loss = 0.0

    print("Finished training")
    print("==========================")
    print("")
    
    os.makedirs("outputs", exist_ok=True)
    file_path = "outputs/cifar_net.pt"
    torch.save(net.state_dict(), file_path) # Anything written to the outputs folder on remote compute is automatically uploaded to the run outputs 
    # run.upload_file(name=file_path, path_or_stream=file_path)
    
    print("Saved and uploaded model")
