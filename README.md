# Heartrate_Analysis
This repository contains my implementation on analyzing PPG/ECG signals using HeartPy, heart rate analysis package of Python.

1. Basic Implementation of analyzing the features of the PPG dataset available with the HeartPy library.
2. Machine Learning classifier-based model for extracting heart rate signal segments from existing or realtime PPG (photoplethysmogram) measurements


PPG signal feature extraction details:

1. The classiﬁcation algorithm designed can be aimed to automatically detect regions in the PPG that contain high quality heart rate components (i.e. high signal
to-noise ratio) for further signal processing and heart rate estimation. One of the ways to distinguish heart rate signal from motion artifacts and noise
in the PPG is by estimating the power spectral density (PSD) of the signal.

2. A distinction of the heart rate signal is its relatively sharp transition, and high ratio, between magnitudes of signal components in the expected heart rate frequency range (1-2Hz) and those in higher frequencies (>3Hz), as opposed to signal segments showing a high degree of randomness due to motion artifacts and noise. Hence, the second feature used is the slope of the power spectral density curves between the low and high frequency component ranges (LF/HF).

3. Prominent peaks are indicative of heart beats, and the distance between them is near-constant in short signal windows under normal physiological conditions. Thus, the regularity, or variance, of inter-peak distance can be used as a discriminating feature between high and low-HR content signal segments. 


