# app.py

from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained multi-class model (cats, dogs, birds)
model = load_model('model/cat_dog_bird_classifier.keras')

UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the classes in the same order as training labels
class_names = ['Bird', 'Cat', 'Dog']

def predict_image(img_path):
    # Adjust target_size if your model expects a different image dimension
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Model outputs probabilities for each class
    prediction = model.predict(img_array)
    # e.g., prediction might look like [[0.2, 0.5, 0.3]]
    
    # Get the class with the highest probability
    class_idx = np.argmax(prediction, axis=1)[0]
    return class_names[class_idx]

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
