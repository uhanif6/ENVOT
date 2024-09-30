# ENVOT: Anomaly Detection in IoT Devices Using Ensemble classifier and Voting mechanism
This repository contains the implementation of ENVOT (Ensemble and Variational Autoencoder-based IoT anomaly detection). ENVOT is a framework designed for anomaly detection in IoT devices, using a combination of Variational Autoencoder (VAE) for feature extraction and an ensemble of RandomForest and XGBoost classifiers for anomaly prediction.

## Introduction
ENVOT is built to detect anomalies in IoT devices, targeting various types of potential attacks like firmware logic errors, parameter tampering, and sensor data manipulation. The project includes both hardware-based testing (Arduino devices) and a simulation environment for large-scale testing using NS3.

## Key Features:
- Variational Autoencoder (VAE) for feature extraction.
- Ensemble model with RandomForest and XGBoost for classification.
 -Supports both real-time and simulated environments for anomaly detection.
- Handles attacks such as firmware logic errors, sensor tampering, and more.

## Dataset
The performance of ENVOT relies on high-quality data from IoT devices, which includes sensor readings and system metrics. Below, we outline how to use the dataset to train and test the model.
- The dataset contains raw memory data extracted from IoT devices, along with labels indicating normal and anomalous behavior.
You can download the dataset used for training and testing the model here: [Dataset Link](https://dx.doi.org/10.21227/hfcg-kj85)

## Requirements
The following dependencies are required to run the project:
```bash
Python 3.7+
TensorFlow
Scikit-learn
keras
imblearn
mpl_toolkits.mplot3d
XGBoost
NS3 (for simulation environment)
Pandas
NumPy
Matplotlib
```
Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/ENVOT.git
cd ENVOT
pip install -r requirements.txt
```

## Project Structure
```bash
ENVOT/
│
├── data/                   # Data files (both training and test datasets)
├── models/                 # Pre-trained models and checkpoints
├── scripts/
│   ├── train_model.py      # Script to train VAE and Ensemble models
│   ├── predict_anomalies.py # Script to predict anomalies on test data
│   ├── feature_extraction.py # Feature extraction methods
│   └── utils.py            # Utility functions
│
├── simulation/             # NS3 simulation code
│   ├── run_simulation.py   # Script to run the IoT simulation
│   └── ns3-config/         # Configuration files for NS3
│
├── figures/                # Generated plots and diagrams
├── README.md               # This file
└── requirements.txt        # Python dependencies
```
Example Commands
- **Training the Model**:
```bash
python scripts/train_model.py --dataset data/iot_data.csv --output models/trained_model.pkl
```
- **Predicting Anomalies**:
```bash
python scripts/predict_anomalies.py --model models/trained_model.pkl --test data/test_iot_data.csv
```
- **Running NS3 Simulation**:
```bash
./ns3 build
NS_LOG=IoTSimulation=info ./ns3 run scratch/simulation
./NetAnim simulation.xml
```

## Key Performance Metrices
- **Accuracy**: Measures the overall correctness of the model's predictions.
- **Precision**: Ratio of correctly predicted positives to total predicted positives.
- **Recall**: Ratio of correctly predicted positives to all actual positives.
- **F1-Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Provides a detailed breakdown of all categories.
- **Time Consumption (Latency)**: Time taken for processing each input.
- **Energy Consumption**: Computational efficiency of the model.
## Results
The results/ folder will contain the confusion matrix, precision-recall metrics, and plots for model evaluation. You can also visualize feature importance using the following command:

```bash
python scripts/plot_feature_importance.py --model models/trained_model.pkl
```
