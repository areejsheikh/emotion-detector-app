import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model = tf.keras.models.load_model('emotion_model.keras')

# Emotion labels (adjust these based on your model's classes)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((48, 48))  # Adjust size according to your model's input requirements
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Save the file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            # Preprocess the image
            processed_image = preprocess_image(file_path)
            
            # Make prediction
            predictions = model.predict(processed_image)
            
            # Get the predicted emotion and confidence
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            emotion = EMOTIONS[predicted_class]
            
            # Clean up
            os.remove(file_path)
            
            return jsonify({
                'emotion': emotion,
                'confidence': confidence,
                'all_predictions': {
                    emotion: float(conf) 
                    for emotion, conf in zip(EMOTIONS, predictions[0])
                }
            })
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(debug=True) 