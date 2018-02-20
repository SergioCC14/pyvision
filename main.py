import sys
import io
import os

from google.cloud import vision
from google.cloud.vision import types

from flask import Flask
from flask import request
app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/vision", methods=['GET'])
def vision():
    img = request.args.get('img_path')
    # img = request.args.get('img_path')

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # The name of the image file to annotate
    file_name = os.path.join(
        os.path.dirname(__file__),
        img)

    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    print('Labels:')
    for label in labels:
        print(label.description)

    return labels
