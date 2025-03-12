import streamlit as st
import numpy as np
import librosa
import joblib
import os
import librosa.display
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load trained models
model = joblib.load("svm_audio_model_pca_rbf_optimized.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# Load test dataset to display accuracy
X_test = joblib.load("X_test.pkl")  # Load test features
y_test = joblib.load("y_test.pkl")  # Load test labels
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

def preprocess_audio(audio_path):
    """Loads, removes silence, and normalizes audio length."""
    audio_data, sr = librosa.load(audio_path, sr=None)
    
    # Remove silence
    non_silent_intervals = librosa.effects.split(audio_data, top_db=30)
    audio_trimmed = np.concatenate([audio_data[start:end] for start, end in non_silent_intervals])

    if audio_trimmed.size == 0:
        st.error(f"Error: Empty audio after trimming {audio_path}")
        return None, None

    # Normalize duration to 5 seconds
    target_length = 5 * sr  
    if len(audio_trimmed) > target_length:
        audio_trimmed = audio_trimmed[:target_length]  
    else:
        audio_trimmed = np.pad(audio_trimmed, (0, max(0, target_length - len(audio_trimmed))))  

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

def plot_spectrogram(audio_path):
    """Plots waveform and spectrogram."""
    audio_data, sr = librosa.load(audio_path, sr=None)
    
    fig, ax = plt.subplots(2, 1, figsize=(8, 5))

    # Waveform
    ax[0].set_title("Waveform")
    librosa.display.waveshow(audio_data, sr=sr, ax=ax[0])

    # Spectrogram
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    img = librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log', cmap='inferno', ax=ax[1])
    ax[1].set_title("Spectrogram")
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")

    st.pyplot(fig)

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

        st.success(f"**Prediction:** {label} \n\n**Confidence:** {confidence_score:.2f}%")

        # Display the final model's accuracy
        st.info(f"**Optimized Model Accuracy:** {accuracy * 100:.2f}%")

        plot_spectrogram(file_path)
    else:
        st.error("Error processing the audio file!")

# Streamlit UI
st.title("🔊 Audio Impersonation Detection")

st.write("Upload an audio file to check whether it is genuine or fake.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Save uploaded file
    file_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict
    predict_audio(file_path)
