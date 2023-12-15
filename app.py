import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
#!/usr/bin/env python3

import requests
import json

API_TOKEN = "259e85cf2b634a12823195e29b860d62"

app = Flask(__name__)

# Load pre-trained models
full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

@app.route('/full_body_detection', methods=['POST'])
def full_body_detection():
    try:
        # Get image from request
        img_data = request.files['image'].read()
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect full body
        full_bodies = full_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around full bodies
        for (x, y, w, h) in full_bodies:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the result or send it as a response
        cv2.imwrite('full_body_detection_result.jpg', img)
        return jsonify(result='full_body_detection_result.jpg')

    except Exception as e:
        return jsonify(error=str(e))

@app.route('/bmi_calculator', methods=['POST'])
def bmi_calculator():
    try:
        # Get data from request
        data = request.get_json()
        height = data['height']
        weight = data['weight']
        age = data['age']

        # BMI calculation
        bmi = weight / ((height / 100) ** 2)

        return jsonify(bmi=bmi)

    except Exception as e:
        return jsonify(error=str(e))

@app.route('/eye_detection', methods=['POST'])
def eye_detection():
    try:
        # Get image from request
        img_data = request.files['image'].read()
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(gray)

        # Draw rectangles around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Save the result or send it as a response
        cv2.imwrite('eye_detection_result.jpg', img)
        return jsonify(result='eye_detection_result.jpg')

    except Exception as e:
        return jsonify(error=str(e))



@app.route('/', methods=['Get'])
def Hello():
    return("server running..")

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    # Check if the uploaded file has an allowed extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return jsonify(error='No file part'), 400

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return jsonify(error='No selected file'), 400

        # Check if the file has an allowed extension
        if file and allowed_file(file.filename):
            # Save the uploaded file to the specified upload folder
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'img.jpg')
            file.save(filename)

            # Return a success response with the uploaded file's name
            image_path = "uploads/img.jpg"
            result = liveness(image_path)
            return jsonify(result)

        return jsonify(error='Invalid file extension'), 400

    except Exception as e:
        return jsonify(error=str(e))
    
def liveness(image_path):
    url = "https://api.luxand.cloud/photo/liveness"
    headers = {"token": API_TOKEN}

    if image_path.startswith("https://"):
        files = {"photo": '/uploads/img.jpg'}
    else:
        files = {"photo": open(image_path, "rb")}

    response = requests.post(url, headers=headers, files=files)
    result = json.loads(response.text)

    if response.status_code == 200:
        return response.json()
    else:
        print("Can't recognize people:", response.text)
        return None

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    

