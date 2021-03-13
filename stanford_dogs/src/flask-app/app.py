# Import libraries
import numpy as np
import os
import requests
import torch
import torchvision.transforms as transforms
from flask import Flask, redirect, render_template, request, send_from_directory, url_for
from PIL import Image
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = set(["jpg", "jpeg", "png"])
UPLOAD_FOLDER = 'uploads'
SCORING_URI = "http://20.73.117.144:80/api/v1/service/dog-classification-service-aks/score"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def preprocess_image(image_file):
    """
    Preprocess an input image.
    :param image_file: Path to the input image
    :return image.numpy(): preprocessed image as numpy array
    """
    
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = Image.open(image_file)
    image = data_transforms(image).float()
    image = image.clone().detach()
    image = image.unsqueeze(0)
    
    return torch.tensor(image.numpy())


def predict(file):
    input_data = preprocess_image(file)

    # Get header and body for POST request
    api_key = 'IpJZFNxfvQQbRhFbiytcXTZhFFs7nQgN'
    input_data = "{\"data\": " + str(input_data.tolist()) + "}"
    headers = {"Content-Type": "application/json", "Authorization":("Bearer "+ api_key)} 

    # Make POST request
    resp = requests.post(SCORING_URI, input_data, headers=headers)
    # img  = load_img(file, target_size=IMAGE_SIZE)
    # img = img_to_array(img)/255.0
    # img = np.expand_dims(img, axis=0)
    # probs = vgg16.predict(img)[0]
    # output = {'Negative:': probs[0], 'Positive': probs[1]}
    return resp.text

app = Flask(__name__, template_folder='Templates')  ## To upload files to folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template("index.html", label="", imagesource="file://null")  ## Routing url


@app.route('/', methods=['GET', 'POST'])  ## Main post and get methods for calling and getting a response from the server
def upload_file(): 
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
    return render_template("index.html", label=output, imagesource=file_path)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(threaded=False)