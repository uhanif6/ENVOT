import sys
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import random
import joblib
import scipy.stats as stats

# Function to extract statistical and frequency features from sequences
def extract_features(sequences, expected_num_features):
    features = []
    for seq in sequences:
        mean = np.mean(seq)
        std = np.std(seq)
        max_val = np.max(seq)
        min_val = np.min(seq)
        median = np.median(seq)
        skewness = stats.skew(seq)
        kurtosis = stats.kurtosis(seq)
        iqr = np.percentile(seq, 75) - np.percentile(seq, 25)
        fft_features = np.abs(np.fft.fft(seq))[:len(seq)//2]  # Use only positive frequencies
        combined_features = [mean, std, max_val, min_val, median, skewness, kurtosis, iqr] + list(fft_features)
        
        # Ensure the number of features is consistent
        if len(combined_features) > expected_num_features:
            combined_features = combined_features[:expected_num_features]
        elif len(combined_features) < expected_num_features:
            combined_features += [0] * (expected_num_features - len(combined_features))

        features.append(combined_features)
    
    return np.array(features)

# Load model and scaler
scaler = joblib.load("scratch/scaler.save")
model = joblib.load("scratch/ensemble_model.pkl")

# Expected number of features (from training)
expected_num_features = scaler.mean_.shape[0]

# Simulate device RAM data by loading files
ram_files = [
    "scratch/anomaly_free_1.txt",
    "scratch/firmware_logic_error_1.txt",
    "scratch/parameter_tampering_1.txt",
    "scratch/sensor_data_manipulation_1.txt"
]

# Randomly select a RAM file for each node
device_id = int(sys.argv[1])
selected_file = random.choice(ram_files)

# Load and preprocess data
with open(selected_file, "r") as file:
    data = [int(x, 16) for x in file.read().split()]
    features = extract_features([data], expected_num_features)
    features = scaler.transform(features)

    # Validate feature length
    if features.shape[1] != expected_num_features:
        raise ValueError(f"Expected {expected_num_features} features, but got {features.shape[1]}.")

# Run the model and print the result
result = model.predict(features)
print(f"Device {device_id} attestation result: {result}")

# Simulate energy consumption (example)
energy_consumed = random.uniform(0.001, 0.002)  # Example energy consumption in mJ
latency = random.uniform(0.5, 3.0)  # Example latency in ms

# Output the energy and latency for the specific device
print(f"Device {device_id} Energy Consumption: {energy_consumed} mJ")
print(f"Device {device_id} Latency: {latency} ms")
