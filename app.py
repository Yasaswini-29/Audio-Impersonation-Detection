import streamlit as st
import numpy as np
import librosa
import joblib
import os
import librosa.display
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# Load trained models
model = joblib.load("svm_audio_model_pca_rbf_optimized.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# Define preprocess_audio function
def preprocess_audio(audio_path):
    """Loads, removes silence, and normalizes audio length."""
    audio_data, sr = librosa.load(audio_path, sr=None)
    
    # Remove silence
    non_silent_intervals = librosa.effects.split(audio_data, top_db=30)
    audio_trimmed = np.concatenate([audio_data[start:end] for start, end in non_silent_intervals])

    if audio_trimmed.size == 0:
        print(f"Error: Empty audio after trimming {audio_path}")
        return None, None

    # Normalize duration to 5 seconds (arbitrary choice)
    target_length = 5 * sr  # 5 seconds
    if len(audio_trimmed) > target_length:
        audio_trimmed = audio_trimmed[:target_length]  # Truncate
    else:
        audio_trimmed = np.pad(audio_trimmed, (0, max(0, target_length - len(audio_trimmed))))  # Pad

    return audio_trimmed, sr

# Define feature extraction function
def extract_features(audio_path, n_mfcc=40, n_fft=2048, hop_length=512):
    """Extracts MFCC, Spectrogram, and Spectral Features."""
    audio_data, sr = preprocess_audio(audio_path)

    if audio_data is None:
        return None

    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(audio_data)

        features = np.hstack((
            np.mean(mfccs, axis=1), np.mean(delta_mfccs, axis=1), np.mean(chroma, axis=1),
            np.mean(mel_spec_db, axis=1), np.mean(spectral_contrast, axis=1),
            np.mean(spectral_rolloff, axis=1), np.mean(zero_crossing, axis=1)
        ))

        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Define prediction function
def predict_audio(file_path):
    """Predicts whether the audio is real or fake and visualizes spectrogram."""
    features = extract_features(file_path)
    
    if features is not None:
        features_scaled = scaler.transform([features])
        features_pca = pca.transform(features_scaled)
        prediction = model.predict(features_pca)[0]
        confidence = model.predict_proba(features_pca)[0]

        label = "🟢 Genuine" if prediction == 1 else "🔴 Fake"
        confidence_score = max(confidence) * 100
        
        return label, confidence_score
    else:
        return None, None

# Streamlit UI
st.title("🎙️ Audio Impersonation Detection")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save uploaded file temporarily
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Make prediction
    label, confidence = predict_audio(file_path)

    if label:
        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2f}%")

    # Show spectrogram
    def plot_spectrogram(audio_path):
        """Plots waveform and spectrogram."""
        audio_data, sr = librosa.load(audio_path, sr=None)
        
        plt.figure(figsize=(10, 5))

        # Waveform
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(audio_data, sr=sr)
        plt.title("Waveform")

        # Spectrogram
        plt.subplot(2, 1, 2)
        spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log', cmap='inferno')
        plt.title("Spectrogram")
        plt.colorbar()

        st.pyplot(plt)

    plot_spectrogram(file_path)

# Display model accuracy on test set
st.subheader("📊 Model Performance")

# Load test set to display accuracy
if "X_test.npy" in os.listdir() and "y_test.npy" in os.listdir():
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Fake", "Genuine"])

    st.write(f"**Test Set Accuracy:** {accuracy * 100:.2f}%")
    st.text("Classification Report:")
    st.text(report)
else:
    st.warning("Test data not found. Please run the training script to generate `X_test.npy` and `y_test.npy`.")

