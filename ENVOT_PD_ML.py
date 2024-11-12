import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier, plot_importance
from imblearn.over_sampling import SMOTE
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import pickle
from tensorflow.keras import layers, models
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
from math import pi
from keras.models import load_model
import joblib

# Function to load and extract critical part of the files
def load_data(file_paths, fraction=0.1):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            content = file.read().split()
            critical_length = int(len(content) * fraction)
            critical_part = content[:critical_length]
            data.append([int(x, 16) for x in critical_part])  # Convert hex to int
    return data

# Function to pad sequences to the same length
def pad_sequences(sequences, maxlen):
    return np.array([seq + [0] * (maxlen - len(seq)) for seq in sequences])

# Function to extract statistical and frequency features from sequences
def extract_features(sequences):
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
        features.append(combined_features)
    return np.array(features)

# Define file paths
train_files = ["#input files for training, anomaly free, parameter tampering, firmware logic error, and sensor data manipulation"]
# Updated test_files with only 10 files
test_files = ["#input files for testing, anomaly free, parameter tampering, firmware logic error, and sensor data manipulation"]

# Updated test_labels for 10 files
test_labels = np.array([0, 3, 2, 0, 1, 1, 3, 0, 3, 2])  # Corresponding labels for test files

# Load training data
train_data = {}
for category, files in train_files.items():
    train_data[category] = load_data(files)

# Find the maximum sequence length
max_len = max(max(len(seq) for seq in data) for data in train_data.values())

# Pad sequences to the same length
for category in train_data:
    train_data[category] = pad_sequences(train_data[category], max_len)

# Load and pad test data
test_data = pad_sequences(load_data(test_files), max_len)

# Extract features from training and test data
train_features = {}
for category in train_data:
    train_features[category] = extract_features(train_data[category])

# Combine all training features and labels
all_train_features = np.concatenate(list(train_features.values()))
train_labels = np.array([0]*4 + [1]*4 + [2]*4 + [3]*4)

# Extract features from test data
test_features = extract_features(test_data)

# Standardize the features
scaler = StandardScaler()
all_train_features = scaler.fit_transform(all_train_features)
test_features = scaler.transform(test_features)

# Data augmentation using SMOTE
smote = SMOTE()
all_train_features, train_labels = smote.fit_resample(all_train_features, train_labels)

# Define the classifiers
rf = RandomForestClassifier()
xgb = XGBClassifier()

# Hyperparameter tuning for RandomForest
param_grid_rf = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=3, n_jobs=-1, verbose=2)
grid_search_rf.fit(all_train_features, train_labels)

# Hyperparameter tuning for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, n_jobs=-1, verbose=2)
grid_search_xgb.fit(all_train_features, train_labels)

# Best estimators
best_rf = grid_search_rf.best_estimator_
best_xgb = grid_search_xgb.best_estimator_

# Ensemble model with weighted voting
ensemble = VotingClassifier(estimators=[('rf', best_rf), ('xgb', best_xgb)], voting='soft', weights=[1, 2])
ensemble.fit(all_train_features, train_labels)

# Predict and evaluate on the test data
test_predictions = ensemble.predict(test_features)
accuracy = accuracy_score(test_labels, test_predictions)

# Show classification report and confusion matrix
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(test_labels, test_predictions, target_names=train_files.keys()))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(test_labels, test_predictions)
print(conf_matrix)

# Print the predictions for each test file
print("\nPredicted categories for each test file:")
for i, file in enumerate(test_files):
    category = list(train_files.keys())[test_predictions[i]]
    print(f"{file}: {category}")

# Combine RandomForest and XGBoost feature importances
importances_rf = best_rf.feature_importances_
importances_xgb = best_xgb.feature_importances_
importances_combined = (importances_rf + importances_xgb) / 2
indices_combined = np.argsort(importances_combined)[::-1]

# Limit the number of features shown in the plot
n_features = 20  # Show top 20 features

# Create a dataframe for visualization
df_combined = pd.DataFrame({
    'Feature Index': indices_combined[:n_features],
    'RandomForest Importance': importances_rf[indices_combined[:n_features]],
    'XGBoost Importance': importances_xgb[indices_combined[:n_features]]
})


# Create a FontProperties object for bold text
bold_font = FontProperties(weight='bold')

# Plot scatter plot
plt.figure(figsize=(14, 8))
plt.scatter(df_combined['Feature Index'], df_combined['RandomForest Importance'], color='b', label='RandomForest Importance', s=100)
plt.scatter(df_combined['Feature Index'], df_combined['XGBoost Importance'], color='g', label='XGBoost Importance', s=100)

# Title and labels with increased font size and bold font weight
# plt.title("Combined Feature Relevance (RandomForest & XGBoost)", fontsize=16, fontweight='bold')
plt.xlabel('Feature Index', fontsize=14, fontweight='bold')
plt.ylabel('Variable Contribution', fontsize=14, fontweight='bold')
# Set the ticks with bold font weight
plt.xticks(fontsize=11, fontweight='bold')
plt.yticks(fontsize=11, fontweight='bold')

# Modify the legend to make the labels bold
legend = plt.legend(fontsize=14, loc='best', markerscale=1.5, fancybox=True, shadow=True)
for text in legend.get_texts():
    text.set_fontweight('bold')

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=train_files.keys(), yticklabels=train_files.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Set consistent figure size for all radar charts
figsize = (10, 7)

# Plot the significant feature values for each file using radar chart
top_features_indices = indices_combined[:n_features]  # Indices of top features

for i, file in enumerate(test_files):
    values = test_features[i, top_features_indices].tolist()
    values += values[:1]  # Repeat the first value to close the circle
    angles = [n / float(n_features) * 2 * pi for n in range(n_features)]
    angles += angles[:1]

    plt.figure(figsize=figsize)  # Ensure the same figure size
    ax = plt.subplot(111, polar=True)

    # Set consistent axis limits (optional)
    ax.set_ylim(-4, 4)  # You can adjust this depending on your value range

    # Set consistent label sizes
    plt.xticks(angles[:-1], top_features_indices, color='grey', size=10)  # Label size is consistent

    # Plot the data
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)

    # Title and Layout
    plt.title(f"Significant Feature Values for {file}", size=15)
    
    # Ensure consistent tight layout without cutting anything off
    plt.tight_layout(pad=3.0)
    
    # Show the plot
    plt.show()

# Visualization for critical 10% of RAM usage
fig = plt.figure(figsize=(12, 13))
ax = fig.add_subplot(111, projection='3d')
ram_usage = [len(load_data([file])[0]) for file in test_files]
total_ram = [max_len for _ in test_files]
x = np.arange(len(test_files))
y = np.zeros(len(test_files))
dx = np.ones(len(test_files))
dy = np.ones(len(test_files))
dz = ram_usage

ax.bar3d(x, y, np.zeros(len(test_files)), dx, dy, dz, color='b', alpha=0.7)
ax.bar3d(x, y, dz, dx, dy, np.array(total_ram) - np.array(dz), color='r', alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(test_files, rotation=90)
ax.set_xlabel('')
ax.set_ylabel('RAM Usage')
ax.set_zlabel('RAM Size')
ax.set_title('Critical 10-20% of RAM Usage')
ax.legend()
plt.show()

# Save the models and scalers for later use
joblib.dump(ensemble, 'E:/PhD Projects/SEAT/Code/ensemble_model.pkl')
joblib.dump(scaler, 'E:/PhD Projects/SEAT/Code/scaler.save')
