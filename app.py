import streamlit as st
import numpy as np
import librosa
import joblib
import librosa.display
import matplotlib.pyplot as plt
import os

# Load trained models
try:
    model = joblib.load("svm_audio_model_pca_rbf_optimized.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")

    # Extract accuracy only if it's available
    accuracy = getattr(model, "best_score_", None)
    if accuracy:
        accuracy *= 100  # Convert to percentage
except FileNotFoundError:
    st.error("Model files not found! Ensure `svm_audio_model_pca_rbf_optimized.pkl`, `scaler.pkl`, and `pca.pkl` exist.")
    st.stop()

# Streamlit UI
st.title("🎵 Audio Impersonation Detection")
st.write("Upload an audio file to check if it is **Genuine 🟢 or Fake 🔴**.")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

def extract_features(audio_path, n_mfcc=40, n_fft=1024, hop_length=256):
    """Extracts MFCCs and statistical features, making it robust to varying lengths."""
    audio_data, sr = librosa.load(audio_path, sr=None)

    if len(audio_data) < hop_length:
        st.error("Audio is too short for feature extraction!")
        return None

    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        delta_mfccs = librosa.feature.delta(mfccs)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)

        # Extract statistics (mean & std) for consistency across audio lengths
        features = np.hstack((
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(delta_mfccs, axis=1), np.std(delta_mfccs, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(mel_spec, axis=1), np.std(mel_spec, axis=1),
            np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1)
        ))

        return features
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

def predict_audio(file_path):
    """Predicts whether the audio is real or fake."""
    features = extract_features(file_path)

    if features is None:
        st.error("Feature extraction failed. Ensure audio is at least 1 second long.")
        return

    features = np.array(features).reshape(1, -1)  

    # Apply PCA & SVM Model
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    prediction = model.predict(features_pca)[0]
    confidence = model.predict_proba(features_pca)[0]

    label = "🟢 Genuine" if prediction == 1 else "🔴 Fake"
    confidence_score = max(confidence) * 100

    st.success(f"**Prediction:** {label} | **Confidence:** {confidence_score:.2f}%")
    
    if accuracy:
        st.write(f"✅ **Optimized Model Accuracy:** {accuracy:.2f}%")
    else:
        st.write("⚠️ **Model accuracy could not be retrieved.** Ensure it is available in the trained model.")

if uploaded_file is not None:
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(file_path, format="audio/wav")
    predict_audio(file_path)
