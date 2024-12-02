# app.py

from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('model/cat_dog_classifier.h5')

UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_image(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return 'Dog' if prediction[0][0] > 0.5 else 'Cat'

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
            img_file.save(img_path)
            pred = predict_image(img_path)
            return render_template('result.html', prediction=pred, img_path=img_file.filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
