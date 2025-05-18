from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
import json

app = Flask(__name__)
CORS(app)

# Get the absolute path to the model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(os.path.dirname(current_dir), 'cat_dog_model')
model_path = os.path.join(model_dir, 'model.keras')
mapping_path = os.path.join(model_dir, 'class_mapping.json')

# Load the model and class mapping
try:
    model = tf.keras.models.load_model(model_path)
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    print("Model loaded successfully")
    print("Class mapping:", class_mapping)
except Exception as e:
    print(f"Error loading model or mapping: {e}")
    model = None
    class_mapping = None

def preprocess_image(image_bytes):
    # Convert bytes to image
    img = Image.open(io.BytesIO(image_bytes))
    # Resize image to match model's expected sizing
    img = img.resize((224, 224))
    # Convert to numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, 0)
    # Apply MobileNetV2 preprocessing
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or class_mapping is None:
        return jsonify({'error': 'Model or class mapping not loaded'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Get prediction
        prediction = model.predict(processed_image)
        raw_prediction = float(prediction[0][0])
        print(f"Raw prediction value: {raw_prediction}")
        
        # Map the prediction to class names using the loaded mapping
        # Model outputs value between 0 and 1:
        # Values closer to 0 mean class 0 (cats)
        # Values closer to 1 mean class 1 (dogs)
        class_idx = 1 if raw_prediction > 0.5 else 0
        class_name = class_mapping[str(class_idx)]
        confidence = raw_prediction if class_idx == 1 else (1 - raw_prediction)
        
        print(f"Class index: {class_idx}, Class name: {class_name}, Confidence: {confidence}")
        
        return jsonify({
            'prediction': class_name.capitalize(),
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 