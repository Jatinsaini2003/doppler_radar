import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

st.set_page_config(layout="wide")

st.title("🛰️ Doppler Radar Classification System")
st.caption("Real-time Bird vs Drone Detection using Micro-Doppler Signals")

# ----------------------------
# Load Model
# ----------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "model", "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------

st.sidebar.title("Control Panel")

uploaded_file = st.sidebar.file_uploader("Upload Radar CSV", type=["csv"])

step_size = st.sidebar.slider("Step Size", 64, 512, 256)
sampling_rate = st.sidebar.number_input("Sampling Rate (Hz)", value=1000)

start_stream = st.sidebar.button("Start Streaming")

# ----------------------------
# MAIN DISPLAY
# ----------------------------

st.subheader("Streaming Radar Signal")
st.write("Y-axis: Signal Amplitude | X-axis: Sample Index (Time)")

wave_chart = st.line_chart()

st.subheader("Prediction Timeline (Drone=1, Bird=0)")
pred_chart = st.line_chart()

result_box = st.empty()
confidence_bar = st.empty()
final_decision_box = st.empty()
freq_box = st.empty()

spec_placeholder = st.empty()

# ----------------------------
# REAL-TIME STREAMING LOGIC
# ----------------------------

if uploaded_file and start_stream:

    df = pd.read_csv(uploaded_file, header=None)
    signal = df.iloc[0].values

    buffer = []
    prediction_history = []

    for i in range(0, len(signal), step_size):

        chunk = signal[i:i+step_size]

        # ------------------------
        # Update buffer
        # ------------------------
        buffer.extend(chunk)

        if len(buffer) > 5000:
            buffer = buffer[-5000:]

        # ------------------------
        # Update waveform
        # ------------------------
        wave_chart.add_rows(pd.DataFrame(chunk))

        # ------------------------
        # Prediction
        # ------------------------
        if len(buffer) == 5000:

            sample = np.array(buffer)

            prediction = model.predict([sample])[0]
            probs = model.predict_proba([sample])[0]
            confidence = np.max(probs)

            prediction_history.append(1 if prediction == 1 else 0)

            # Timeline update
            pred_chart.add_rows(pd.DataFrame([prediction_history[-1]]))

            # Display prediction
            if prediction == 1:
                result_box.success(f"🚁 DRONE | {confidence*100:.1f}% confidence")
            else:
                result_box.info(f"🦅 BIRD | {confidence*100:.1f}% confidence")

            # Confidence bar
            confidence_bar.progress(int(confidence * 100))

            # ------------------------
            # Stability (majority voting)
            # ------------------------
            if len(prediction_history) >= 10:
                recent = prediction_history[-10:]
                final = max(set(recent), key=recent.count)

                final_label = "🚁 DRONE" if final == 1 else "🦅 BIRD"
                final_decision_box.write(f"### Stable Decision: {final_label}")

            # ------------------------
            # Spectrogram
            # ------------------------
            f, t, Sxx = spectrogram(sample, fs=sampling_rate)

            fig, ax = plt.subplots()

            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")

            ax.pcolormesh(
                t,
                f,
                10 * np.log10(Sxx + 1e-10),
                shading="gouraud",
                cmap="inferno"
            )

            ax.set_title("Micro-Doppler Spectrogram", color="white")
            ax.set_xlabel("Time", color="white")
            ax.set_ylabel("Frequency (Hz)", color="white")

            ax.tick_params(colors="white")

            spec_placeholder.pyplot(fig)

            # ------------------------
            # Dominant Frequency
            # ------------------------
            dominant_freq = f[np.argmax(np.mean(Sxx, axis=1))]
            freq_box.write(f"Dominant Doppler Frequency: {dominant_freq:.2f} Hz")

        time.sleep(0.05)