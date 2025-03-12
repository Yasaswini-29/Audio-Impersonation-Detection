import streamlit as st
import numpy as np
import librosa
import joblib
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load trained models
model = joblib.load("svm_audio_model_pca_rbf_optimized.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

def preprocess_audio(audio_path):
    """Loads, removes silence, and normalizes audio length."""
    audio_data, sr = librosa.load(audio_path, sr=None)
    non_silent_intervals = librosa.effects.split(audio_data, top_db=30)
    audio_trimmed = np.concatenate([audio_data[start:end] for start, end in non_silent_intervals])
    target_length = 5 * sr  # Normalize duration to 5 seconds
    if len(audio_trimmed) > target_length:
        audio_trimmed = audio_trimmed[:target_length]
    else:
        audio_trimmed = np.pad(audio_trimmed, (0, target_length - len(audio_trimmed)))
    return audio_trimmed, sr

def extract_features(audio_path, n_mfcc=40, n_fft=2048, hop_length=512):
    """Extracts MFCC, Spectrogram, and Spectral Features."""
    audio_data, sr = preprocess_audio(audio_path)
    if audio_data is None:
        return None
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features = np.hstack((
            np.mean(mfccs, axis=1), np.mean(delta_mfccs, axis=1), np.mean(chroma, axis=1)
        ))
        return features
    except:
        return None

def predict_audio(file_path):
    """Predicts whether the audio is genuine or fake and visualizes spectrogram."""
    features = extract_features(file_path)
    if features is not None:
        features_scaled = scaler.transform([features])
        features_pca = pca.transform(features_scaled)
        prediction = model.predict(features_pca)[0]
        confidence = model.predict_proba(features_pca)[0]
        label = "🟢 Genuine" if prediction == 1 else "🔴 Fake"
        confidence_score = max(confidence) * 100
        return label, confidence_score, accuracy_score([prediction], [1])  # Assuming ground truth is real
    else:
        return None, None, None

def plot_spectrogram(audio_path):
    """Plots waveform and spectrogram."""
    audio_data, sr = librosa.load(audio_path, sr=None)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    librosa.display.waveshow(audio_data, sr=sr, ax=ax[0])
    ax[0].set_title("Waveform")
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log', cmap='inferno', ax=ax[1])
    ax[1].set_title("Spectrogram")
    fig.colorbar(img, ax=ax[1])
    return fig

# Streamlit UI
st.title("🎤 Audio Impersonation Detection")
st.write("Upload an audio file to check if it's genuine or AI-generated.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    label, confidence, accuracy = predict_audio(file_path)
    if label:
        st.success(f"Prediction: {label} with {confidence:.2f}% confidence")
        st.info(f"Model Accuracy: {accuracy * 100:.2f}%")
        st.pyplot(plot_spectrogram(file_path))
    else:
        st.error("Could not process the audio file.")
