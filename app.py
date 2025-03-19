import streamlit as st
import numpy as np
import librosa
import joblib
import os
import librosa.display
import matplotlib.pyplot as plt

# Load trained models and scalers
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
model = joblib.load("svm_audio_model_pca_rbf_optimized.pkl")

# Feature extraction function
def extract_features(audio_path, n_mfcc=100, n_fft=2048, hop_length=512):
    try:
        # Load audio (Removing 'backend' argument)
        audio_data, sr = librosa.load(audio_path, sr=None)

        if audio_data.size == 0:
            print("Error: Empty audio file.")
            return None

        # Extract features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=n_fft, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(audio_data)
        spec, _ = librosa.magphase(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length))
        spec_mean = np.mean(spec, axis=1)

        features = np.hstack((
            np.mean(mfccs, axis=1), np.mean(delta_mfccs, axis=1), np.mean(chroma, axis=1),
            np.mean(mel_spec_db, axis=1), np.mean(spectral_contrast, axis=1),
            np.mean(spectral_rolloff, axis=1), np.mean(zero_crossing, axis=1), spec_mean
        ))

        return features
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None


# Function to plot spectrogram
def plot_spectrogram(audio_path):
    audio_data, sr = librosa.load(audio_path, sr=None)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    # Waveform Plot
    ax[0].set_title("Waveform")
    librosa.display.waveshow(audio_data, sr=sr, ax=ax[0])

    # Spectrogram Plot
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(spec, sr=sr, x_axis="time", y_axis="log", cmap="inferno", ax=ax[1])
    ax[1].set_title("Spectrogram")
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")

    return fig

# Function to predict audio authenticity
def predict_audio(file_path):
    features = extract_features(file_path)
    if features is not None:
        features_scaled = scaler.transform([features])
        features_pca = pca.transform(features_scaled)
        prediction = model.predict(features_pca)[0]
        confidence = model.predict_proba(features_pca)[0]

        label = "🟢 Genuine" if prediction == 1 else "🔴 Fake"
        confidence_score = max(confidence) * 100
        return label, confidence_score
    return None, None

# Streamlit UI
st.title("🎤 Audio Impersonation Detection")
st.write("Upload an audio file to check if it's genuine or fake.")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save uploaded file temporarily
    file_path = f"temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Make prediction
    label, confidence = predict_audio(file_path)

    if label:
        st.subheader(f"Prediction: {label}")
        st.subheader(f"Confidence: {confidence:.2f}%")

        # Show spectrogram
        st.pyplot(plot_spectrogram(file_path))

    # Clean up temp file
    os.remove(file_path)
