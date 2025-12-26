# EEG-seizure-detection-DNN

the goal is to seizure event detection from multichannel EEG recordings by transforming raw data (biomedical signals) into meaningful features and training a supervised classifier.

the pipeline processes raw EEG (EDF format) extracts frequency-domain features using bandpower analysis, assigns window-level seizure labels, and evaluates a classification model using robust statistical metrics.


#📌 Project Objectives
Load and preprocess raw EEG signals

Segment EEG into overlapping temporal windows

Extract physiologically meaningful frequency features

Label windows based on seizure intervals

Train and evaluate a seizure detection model

Visualize signals, features, and performance metrics


#🔬 Dataset
Source: CHB-MIT Scalp EEG Database

Format: EDF (European Data Format)

Sampling rate: ~256 Hz

Channels: 23 EEG channels

Annotations: Seizure start and end times

⚠️ The dataset is not included in this repository due to licensing restrictions.


#⚙️ Methods
1. Signal Processing
Bandpass filtering (0.5–40 Hz)

Sliding window segmentation (10s windows, 5s step)

Frequency band integration:

Delta (0.5–4 Hz)

Theta (4–8 Hz)

Alpha (8–13 Hz)

Beta (13–30 Hz)


2. Feature Engineering
Power spectral density (Welch method)

Bandpower per channel and frequency band

Flattened into feature vectors per window


3. Labeling
Windows overlapping with seizure intervals are labeled as positive

Binary classification (seizure vs non-seizure)


4. Modeling
Logistic Regression classifier

Stratified train/test split

Evaluation metrics beyond accuracy due to class imbalance

#📊 Results
Metric                Value

ROC AUC               ~0.91

PR AUC                ~0.46

Sensitivity (Recall+) ~0.67

Accuracy              ~0.92


#📈 Visualizations
Raw EEG signal plots

Bandpower distributions

ROC and Precision-Recall curves

Confusion matrix


#🧪 Statistical Analysis
Welch’s t-test for feature comparison

Multiple hypothesis correction (FDR)

Feature importance via model coefficients


#📚 Technologies Used
Python

NumPy / SciPy

MNE

Scikit-learn

Statsmodels

Matplotlib


#🚀 Future Improvements
Deep learning models (CNN/LSTM on raw signals)

Multi-patient generalization

Class imbalance mitigation (SMOTE, focal loss)

Temporal models with sequential context


#👤 Author

Harpreet Singh

Aspiring ML Engineer in NeuroAI

GitHub: https://github.com/harleone




