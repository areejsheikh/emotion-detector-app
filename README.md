# Emotion Detector Web Application

A professional web application that uses a deep learning model to detect emotions in images. The application provides a modern, user-friendly interface for uploading images and viewing emotion predictions.

## Features

- Drag and drop image upload
- Real-time image preview
- Detailed emotion predictions with confidence scores
- Modern, responsive design using Tailwind CSS
- Loading indicators and error handling

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Click the upload area or drag and drop an image
2. Preview your image
3. Click "Detect Emotion" to analyze
4. View the results showing:
   - Primary detected emotion
   - Confidence scores for all emotions
   - Visual confidence bars

## Technical Details

- Built with Flask
- Uses TensorFlow for emotion detection
- Frontend styled with Tailwind CSS
- Supports common image formats (JPEG, PNG)
- Processes images in real-time

## Requirements

- Python 3.8+
- See requirements.txt for full dependencies 