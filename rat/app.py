from flask import Flask, request, jsonify, send_from_directory
import os
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from gtts import gTTS
import librosa  

app = Flask(__name__)

# Load Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Directory for saving files
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_speech'

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
allowed_extensions = {'mp3', 'wav', 'flac', 'ogg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Define the mapping of accent selection to language code
ACCENT_VOICES = {
    '0': 'en',  # American English 
    '1': 'en-uk',  # British English
    '2': 'en-in',  # Indian English
    '3': 'en-au',  # Australian English
}

# Serve the index.html file
@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    accent = request.form.get('accent', '')  # Get accent from form data

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only audio files are allowed."}), 400
    
    # Save and process the file here
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Process the audio file here, transcribe it and generate translated speech
    transcription = transcribe_audio(file_path)
    if "error" in transcription:
        return jsonify({"error": transcription}), 500
    
    translated_speech = generate_speech_from_text(transcription, accent)
    if "error" in translated_speech:
        return jsonify({"error": translated_speech}), 500

    # Return the URL for the translated audio file
    translated_audio_url = f"/audio/{os.path.basename(translated_speech)}"
    return jsonify({"message": "File uploaded and processed successfully!", "audio_url": translated_audio_url}), 200


def transcribe_audio(audio_path):
    try:
        # Load the audio file for transcription
        audio, rate = librosa.load(audio_path, sr=16000)
        input_values = processor(audio, sampling_rate=rate, return_tensors="pt").input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription
    except Exception as e:
        return str(e)


def generate_speech_from_text(text, accent):
    try:
        # Get the correct language code for the selected accent
        lang_code = ACCENT_VOICES.get(accent, 'en')  # Default to 'en' if no valid accent is selected
        
        # Use gTTS for accent-specific speech generation
        tts = gTTS(text=text, lang=lang_code)
        output_filename = os.path.join(OUTPUT_FOLDER, "translated_speech.mp3")
        tts.save(output_filename)
        
        return output_filename
    except Exception as e:
        return str(e)


@app.route('/audio/<filename>')
def get_audio_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
