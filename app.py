import streamlit as st
import numpy as np
import librosa
import joblib
import librosa.display
import matplotlib.pyplot as plt

# Load trained models
try:
    model = joblib.load("svm_audio_model_pca_rbf_optimized.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")

    # Extract model accuracy if available
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

def extract_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extracts **exactly** 229 features using MFCCs, chroma, mel spectrogram, spectral contrast.
    The final feature vector contains mean and std values to ensure consistency.
    """
    audio_data, sr = librosa.load(audio_path, sr=None)

    if len(audio_data) < hop_length:
        st.error("Audio is too short for feature extraction!")
        return None

    try:
        # **Extract Features**
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        delta_mfccs = librosa.feature.delta(mfccs)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)

        # **Feature Selection & Summarization (Mean + Std)**
        features = np.hstack((
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(delta_mfccs, axis=1), np.std(delta_mfccs, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(mel_spec, axis=1), np.std(mel_spec, axis=1),
            np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1)
        ))

        # **Ensure Correct Feature Size**
        if features.shape[0] != 229:
            st.error(f"Feature extraction mismatch: Got {features.shape[0]} features, expected 229.")
            return None

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
    try:
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

    except ValueError as e:
        st.error(f"Prediction error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

if uploaded_file is not None:
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(file_path, format="audio/wav")
    predict_audio(file_path)
