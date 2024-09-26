# ENVOT: Anomaly Detection in IoT Devices Using Ensemble and VAE
This repository contains the implementation of ENVOT (Ensemble and Variational Autoencoder-based IoT anomaly detection). ENVOT is a framework designed for anomaly detection in IoT devices, using a combination of Variational Autoencoder (VAE) for feature extraction and an ensemble of RandomForest and XGBoost classifiers for anomaly prediction.

## Introduction
ENVOT is built to detect anomalies in IoT devices, targeting various types of potential attacks like firmware logic errors, parameter tampering, and sensor data manipulation. The project includes both hardware-based testing (Arduino devices) and a simulation environment for large-scale testing using NS3.

## Key Features:
- Variational Autoencoder (VAE) for feature extraction.
- Ensemble model with RandomForest and XGBoost for classification.
 -Supports both real-time and simulated environments for anomaly detection.
- Handles attacks such as firmware logic errors, sensor tampering, and more.
## Requirements
The following dependencies are required to run the project:
```bash
Python 3.7+
TensorFlow
Scikit-learn
XGBoost
NS3 (for simulation environment)
Pandas
NumPy
Matplotlib
Installation
```
Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/ENVOT.git
cd ENVOT
pip install -r requirements.txt
```
## Usage
Training and Running the Model
- **Data Preparation**: Ensure that the dataset is in the appropriate format (e.g., hex to integers for RAM data). The critical 10% of the data is processed during this phase.
- **Model Training**: Run the script to train the VAE model and the ensemble classifiers (RandomForest and XGBoost).
```bash
python train_model.py --dataset path/to/dataset
```
- **Anomaly Detection**: After training, you can predict anomalies using the test dataset.
```bash
python predict_anomalies.py --model path/to/saved_model --test path/to/test_data
```
- **Running Simulation (NS3)**
To run the NS3 simulation for large-scale IoT devices, follow the instructions provided in the simulation/README.md.

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
cd simulation
python run_simulation.py
```
## Results
The results/ folder will contain the confusion matrix, precision-recall metrics, and plots for model evaluation. You can also visualize feature importance using the following command:

```bash
python scripts/plot_feature_importance.py --model models/trained_model.pkl
```
