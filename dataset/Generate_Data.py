import numpy as np
import pandas as pd

SIGNAL_LENGTH = 5000

def generate_real_like_signal():

    t = np.linspace(0, 1, SIGNAL_LENGTH)

    # multiple mixed frequencies (irregular)
    signal = (
        0.6*np.sin(2*np.pi*np.random.uniform(5,15)*t) +
        0.4*np.sin(2*np.pi*np.random.uniform(15,30)*t) +
        0.3*np.sin(2*np.pi*np.random.uniform(1,5)*t)
    )

    # random amplitude variation
    envelope = 1 + 0.5*np.random.normal(0, 0.5, SIGNAL_LENGTH)

    signal = signal * envelope

    # strong noise (important!)
    noise = np.random.normal(0, 0.3, SIGNAL_LENGTH)

    signal = signal + noise

    # 🔥 VERY IMPORTANT: center around 0
    signal = signal - np.mean(signal)

    return signal


data = [generate_real_like_signal() for _ in range(10)]

pd.DataFrame(data).to_csv("real_like_signals.csv", index=False)

print("Generated correctly!")