import streamlit as st
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import librosa.display
import os

# Load trained models
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
model = joblib.load("svm_audio_model_pca_rbf_optimized.pkl")

def extract_features(audio_path, n_mfcc=100, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
        if audio_data.size == 0:
            st.error("Error: Empty audio file.")
            return None

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
        st.error(f"Error processing audio: {e}")
        return None

def plot_spectrogram(audio_path):
    """Plots waveform and spectrogram of the audio file."""
    audio_data, sr = librosa.load(audio_path, sr=None)
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    
    # Waveform Plot
    ax[0].set_title("Waveform")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")
    librosa.display.waveshow(audio_data, sr=sr, ax=ax[0])
    
    # Spectrogram Plot
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log', cmap='inferno', ax=ax[1])
    ax[1].set_title("Spectrogram")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    
    st.pyplot(fig)

def predict_audio(audio_path):
    """Predicts whether the audio is real or fake and visualizes the spectrogram."""
    features = extract_features(audio_path)
    
    if features is not None:
        features_scaled = scaler.transform([features])
        features_pca = pca.transform(features_scaled)
        prediction = model.predict(features_pca)[0]
        confidence = model.predict_proba(features_pca)[0]
        
        label = "🟢 Genuine" if prediction == 1 else "🔴 Fake"
        confidence_score = max(confidence) * 100
        
        st.success(f"Prediction: {label}")
        st.write(f"Confidence Score: {confidence_score:.2f}%")
        
        plot_spectrogram(audio_path)
    else:
        st.error("Could not process the audio file.")

# Streamlit UI
st.title("🔍 Audio Impersonation Detection")
st.write("Upload an audio file to check if it's genuine or fake.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    file_path = os.path.join("temp_audio.wav")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(file_path, format='audio/wav')
    predict_audio(file_path)
