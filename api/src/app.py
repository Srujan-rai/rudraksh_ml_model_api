from flask import Flask, jsonify, request
from flask_cors import CORS
from torch.functional import split
from model_files.ml_predict import predict_plant, Network
from pyfcm import FCMNotification
import base64
from decouple import config
import os

app = Flask("Plant Disease Detector")
CORS(app)

# push_service = FCMNotification(api_key=config('FCM_API_KEY'))

@app.route('/', methods=['POST'])
def predict():
    key_dict = request.get_json()
    image = key_dict["image"]
    imgdata = base64.b64decode(image)
    model = Network()
    result, remedy = predict_plant(model, imgdata)
    plant = result.split("___")[0]
    disease = " ".join((result.split("___")[1]).split("_"))
    response = {
        "plant": plant,
        "disease": disease,
        "remedy": remedy,
    }
    response = jsonify(response)
    return response

if __name__ == '__main__':
    # Use Gunicorn as the web server
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 8080))  # Use the PORT environment variable provided by Render
    app.run(debug=True, host=host, port=port)
