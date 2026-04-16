import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

st.set_page_config(layout="wide")

st.title("🛰️ Micro-Doppler Radar Classification System")
st.write("Drone vs Bird Detection using Machine Learning")

# Load model safely
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "model", "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(model_path, "rb") as f:
    model = pickle.load(f)

uploaded_file = st.file_uploader("Upload Radar Signal CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, header=None)

    signal = df.iloc[0, :5000].values

    st.subheader("Streaming Radar Signal")

    chart = st.line_chart()

    col1, col2 = st.columns(2)

    result_box = col1.empty()
    spec_placeholder = col2.empty()

    buffer = []

    for value in signal:

        buffer.append(value)

        if len(buffer) > 5000:
            buffer.pop(0)

        # update waveform
        chart.add_rows(pd.DataFrame([value]))

        # run prediction when buffer full
        if len(buffer) == 5000:

            prediction = model.predict([buffer])[0]
            probs = model.predict_proba([buffer])[0]

            confidence = np.max(probs)

            if prediction == 1:
                result_box.success(f"🚁 Drone Detected | Confidence: {confidence:.2f}")
            else:
                result_box.info(f"🦅 Bird Detected | Confidence: {confidence:.2f}")

            # Generate spectrogram
            f, t, Sxx = spectrogram(np.array(buffer), fs=1000)

            fig, ax = plt.subplots()
            ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (sec)')
            ax.set_title('Micro-Doppler Spectrogram')

            spec_placeholder.pyplot(fig)

        time.sleep(0.005)