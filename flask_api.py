
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from keras.models import load_model
import numpy as np
from PIL import Image
from gtts import gTTS
import os
import time
import traceback

app = Flask(__name__)
CORS(app)

# Ensure audio directory exists
if not os.path.exists("audio_outputs"):
    os.makedirs("audio_outputs")

# Load model
MODEL_PATH = "traffic_classifier.h5"
try:
    print(f"📦 Loading model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# Traffic sign classes
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'No passing', 8: 'No passing veh over 3.5 tons',
    9: 'Right-of-way at intersection', 10: 'Priority road', 11: 'Yield', 12: 'Stop',
    13: 'No vehicles', 14: 'Veh > 3.5 tons prohibited', 15: 'No entry', 16: 'General caution',
    17: 'Dangerous curve left', 18: 'Dangerous curve right', 19: 'Double curve',
    20: 'Bumpy road', 21: 'Slippery road', 22: 'Road narrows on the right',
    23: 'Road work', 24: 'Traffic signals', 25: 'Pedestrians', 26: 'Children crossing',
    27: 'Bicycles crossing', 28: 'Beware of ice/snow', 29: 'Wild animals crossing',
    30: 'End speed + passing limits', 31: 'Turn right ahead', 32: 'Turn left ahead',
    33: 'Ahead only', 34: 'Go straight or right', 35: 'Go straight or left',
    36: 'Keep right', 37: 'Keep left', 38: 'Roundabout mandatory',
    39: 'End of no passing', 40: 'End no passing veh > 3.5 tons'
}

@app.route('/classify', methods=['POST'])
def classify():
    print("📥 classify() endpoint was hit")

    if 'image' not in request.files:
        print("❌ No image in request")
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image_file = request.files['image']
        print("📸 File received:", image_file.filename)

        img = Image.open(image_file).convert("RGB").resize((30, 30))
        img_array = np.expand_dims(np.array(img), axis=0)

        if model is None:
            print("❌ Model is not loaded")
            return jsonify({'error': 'Model not loaded'}), 500

        predictions = model.predict(img_array)[0]
        pred = np.argmax(predictions)
        confidence = float(np.max(predictions))

        print(f"🧠 Prediction Index: {pred} with confidence {confidence:.2f}")

        label = classes.get(pred, "Unknown Sign")
        print(f"✅ Predicted Sign: {label}")

        # Create audio message
        audio_message = f"Be alert, {label} ahead."
        timestamp = int(time.time())
        audio_filename = f"audio_outputs/output_{timestamp}.mp3"

        tts = gTTS(text=audio_message, lang='en')
        tts.save(audio_filename)
        print("🔊 Audio saved:", audio_filename)

        return jsonify({
            'label': label,
            'confidence': round(confidence * 100),
            'audio': f"/audio/{os.path.basename(audio_filename)}"
        })

    except Exception as e:
        print("🔥 Exception occurred:", str(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    path = os.path.join("audio_outputs", filename)
    if os.path.exists(path):
        return send_file(path, mimetype="audio/mpeg")
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
