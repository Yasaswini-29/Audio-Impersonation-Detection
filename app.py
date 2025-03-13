import streamlit as st
import numpy as np
import librosa
import joblib
import os
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

# Load trained models
try:
    model = joblib.load("svm_audio_model_pca_rbf_optimized.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
except FileNotFoundError:
    st.error("Model files not found! Make sure `svm_audio_model_pca_rbf_optimized.pkl`, `scaler.pkl`, and `pca.pkl` exist.")
    st.stop()

st.title("🎵 Audio Impersonation Detection")
st.write("Upload an audio file to check if it is **Genuine 🟢 or Fake 🔴**.")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

def preprocess_audio(audio_path):
    """Loads, removes silence, and normalizes audio length."""
    audio_data, sr = librosa.load(audio_path, sr=None)
    
    # Remove silence
    non_silent_intervals = librosa.effects.split(audio_data, top_db=30)
    audio_trimmed = np.concatenate([audio_data[start:end] for start, end in non_silent_intervals])

    if audio_trimmed.size == 0:
        return None, None

    # Normalize duration to 5 seconds
    target_length = 5 * sr
    if len(audio_trimmed) > target_length:
        audio_trimmed = audio_trimmed[:target_length]
    else:
        audio_trimmed = np.pad(audio_trimmed, (0, max(0, target_length - len(audio_trimmed))))

    return audio_trimmed, sr

def extract_features(audio_path):
    """Extracts features ensuring 229 dimensions to match trained model."""
    audio_data, sr = preprocess_audio(audio_path)

    if audio_data is None:
        return None

    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        delta_mfccs = librosa.feature.delta(mfccs)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(audio_data)

        feature_vector = np.hstack([
            np.mean(mfccs, axis=1), np.mean(delta_mfccs, axis=1),
            np.mean(chroma, axis=1), np.mean(spectral_centroid),
            np.mean(spectral_bandwidth), np.mean(spectral_contrast, axis=1),
            np.mean(spectral_rolloff), np.mean(zero_crossing)
        ])

        # Ensure fixed-size output (229 features)
        expected_features = 229
        if len(feature_vector) < expected_features:
            feature_vector = np.pad(feature_vector, (0, expected_features - len(feature_vector)), mode='constant')
        elif len(feature_vector) > expected_features:
            feature_vector = feature_vector[:expected_features]  # Trim extra values

        return feature_vector
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def plot_spectrogram(audio_path):
    """Plots waveform and spectrogram."""
    audio_data, sr = librosa.load(audio_path, sr=None)
    
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    # Waveform
    librosa.display.waveshow(audio_data, sr=sr, ax=ax[0])
    ax[0].set_title("Waveform")

    # Spectrogram
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log', cmap='inferno', ax=ax[1])
    ax[1].set_title("Spectrogram")
    fig.colorbar(img, ax=ax[1])

    st.pyplot(fig)

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

        st.success(f"**Prediction:** {label} | **Confidence:** {confidence_score}%")

        plot_spectrogram(file_path)
    else:
        st.error("Error processing the audio file!")

if uploaded_file is not None:
    # Save the uploaded file
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Play the uploaded audio
    st.audio(file_path, format="audio/wav")

    # Predict
    predict_audio(file_path)
