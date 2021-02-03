# Import libraries
import argparse
import json
import os
import torch
import torch.nn as nn
from azureml.core.model import Model
from torchvision import transforms


def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path where the images are stored")
    args = parser.parse_args()

    
    
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pt")
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()
    
    image_dataset = {x: datasets.ImageFolder(os.path.join(args.data_path, x),
                                              data_transforms[x])
                      for x in ["train", "val"]}


def run(batch_input_data):
    
    input_data = torch.tensor(json.loads(input_data)["data"])

    # get prediction
    with torch.no_grad():
        output = model(input_data)
        classes = ["chicken", "turkey"]
        softmax = nn.Softmax(dim=1)
        pred_probs = softmax(output).numpy()[0]
        index = torch.argmax(output, 1)

    result = {"label": classes[index], "probability": str(pred_probs[index])}
    return result


    result_list = []
    for file_path in mini_batch:
        test_image = file_to_tensor(file_path)
        out = g_tf_sess.run(test_image)
        result = g_tf_sess.run(probabilities, feed_dict={input_images: [out]})
        result_list.append(os.path.basename(file_path) + ": " + label_dict[result[0]])


def load_data(data_dir):
    """
    Load the train/val data.
    :return (dataloaders, dataset_sizes, class_names):
        dataloaders: dictionary containing pytorch train and validation dataloaders
        dataset_sizes: dictionary containing the size of the training and validation datasets
        class_names: list containing all class names
    """

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ["train", "val"]}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ["train", "val"]}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names
