# Import libraries
import argparse
import json
import numpy as np
import os
import torch
import torch.nn as nn
from azureml.core.model import Model
from PIL import Image
from torchvision import transforms


def init():
    
    global model
    
    parser = argparse.ArgumentParser(description="Start the Pytorch model serving")
    parser.add_argument("--model_name", dest="model_name", required=True)
    args, _ = parser.parse_known_args()
    
    model_path = Model.get_model_path(args.model_name)
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()
    
    
def run(mini_batch):
    
    result_list = []
    for file_path in mini_batch:
        image = preprocess_image(file_path)
        
        # get prediction
        with torch.no_grad():
            output = model(image)
            classes = ["chicken", "turkey"]
            softmax = nn.Softmax(dim=1)
            pred_probs = softmax(output).numpy()[0]
            index = torch.argmax(output, 1)
            result = os.path.basename(file_path) + ", " + str(classes[index]) + ", " + str(pred_probs[index])
            result_list.append(result)
    
    return result_list


def preprocess_image(image_file):
    """
    Preprocess an input image.
    :param image_file: Path to the input image
    :return image: preprocessed image as torch tensor
    """
    
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_file)
    image = data_transforms(image)
    image = image[np.newaxis, ...]
    
    return image
