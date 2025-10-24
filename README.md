# requirements: Flask, fer, opencv-python-headless, numpy
# install with: pip install Flask fer opencv-python-headless numpy

from flask import Flask, request, jsonify
from fer import FER
import cv2
import numpy as np

app = Flask(__name__)
detector = FER()  # Initialize the emotion detector

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read image and convert bytes to NumPy array
    img_bytes = file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Invalid image'}), 400

    # Detect all emotion scores
    emotions = detector.detect_emotions(image)
    # Use top_emotion for summary
    emotion, score = detector.top_emotion(image)

    result = {
        'top_emotion': emotion,
        'confidence': score,
        'all_emotions': emotions
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)