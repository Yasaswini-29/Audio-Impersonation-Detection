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
except FileNotFoundError:
    st.error("Model files not found! Ensure `svm_audio_model_pca_rbf_optimized.pkl`, `scaler.pkl`, and `pca.pkl` exist.")
    st.stop()

st.title("🎵 Audio Impersonation Detection")
st.write("Upload an audio file to check if it is **Genuine 🟢 or Fake 🔴**.")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

def preprocess_audio(audio_path):
    """Loads audio, removes silence, and ensures consistent feature length."""
    audio_data, sr = librosa.load(audio_path, sr=None)
    
    # Remove silence
    non_silent_intervals = librosa.effects.split(audio_data, top_db=30)
    audio_trimmed = np.concatenate([audio_data[start:end] for start, end in non_silent_intervals])

    if audio_trimmed.size == 0:
        return None, None

    # Normalize duration (adaptive padding for short audios)
    min_length = 2 * sr  # 2 seconds minimum
    max_length = 5 * sr  # 5 seconds max

    if len(audio_trimmed) < min_length:
        audio_trimmed = np.pad(audio_trimmed, (0, min_length - len(audio_trimmed)))
    elif len(audio_trimmed) > max_length:
        audio_trimmed = audio_trimmed[:max_length]

    return audio_trimmed, sr

def extract_features(audio_path):
    """Extracts MFCCs, chroma, and other spectral features robustly."""
    audio_data, sr = preprocess_audio(audio_path)

    if audio_data is None:
        return None

    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        delta_mfccs = librosa.feature.delta(mfccs)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(audio_data)

        # Fixed-size feature vector
        features = np.hstack([
            np.mean(mfccs, axis=1), np.mean(delta_mfccs, axis=1),
            np.mean(chroma, axis=1), np.mean(spectral_centroid, axis=1),
            np.mean(spectral_bandwidth, axis=1), np.mean(zero_crossing, axis=1)
        ])

        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def predict_audio(file_path):
    """Predicts whether the audio is real or fake while handling short audios correctly."""
    features = extract_features(file_path)
    
    if features is not None:
        features_scaled = scaler.transform([features])
        features_pca = pca.transform(features_scaled)
        prediction = model.predict(features_pca)[0]

        try:
            confidence = model.predict_proba(features_pca)[0]
            confidence_score = max(confidence) * 100
        except AttributeError:
            confidence_score = "N/A"

        label = "🟢 Genuine" if prediction == 1 else "🔴 Fake"

        duration = librosa.get_duration(filename=file_path)
        if duration < 3:
            st.warning("⚠️ Short audio detected (<3 sec). Prediction may be less accurate.")

        st.success(f"**Prediction:** {label} | **Confidence:** {confidence_score}%")
    else:
        st.error("Error processing the audio file!")

if uploaded_file is not None:
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(file_path, format="audio/wav")

    predict_audio(file_path)

