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
    
    parser = argparse.ArgumentParser(description="Start the Pytorch model serving")
    parser.add_argument('--model_name', dest="model_name", required=True)
    args, _ = parser.parse_known_args()
    
    model_path = Model.get_model_path(args.model_name)
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()


# +
def run(batch_input_data):
    
    return batch_input_data
#     input_data = torch.tensor(json.loads(input_data)["data"])

#     # get prediction
#     with torch.no_grad():
#         output = model(input_data)
#         classes = ["chicken", "turkey"]
#         softmax = nn.Softmax(dim=1)
#         pred_probs = softmax(output).numpy()[0]
#         index = torch.argmax(output, 1)

#     result = {"label": classes[index], "probability": str(pred_probs[index])}
#     return result
