# ENVOT: Anomaly Detection in IoT Devices

## Overview
**ENVOT** (Ensemble-based Variational Autoencoder for IoT) is a framework for anomaly detection in IoT devices. It leverages Variational Autoencoders (VAEs) for feature extraction and an ensemble model combining RandomForest and XGBoost to detect anomalies in IoT device data streams.

The system is designed to work with real IoT devices and large-scale simulations, providing an efficient and scalable solution for IoT anomaly detection.

## Features
- Feature extraction using Variational Autoencoders (VAEs)
- Ensemble model combining RandomForest and XGBoost
- Anomaly detection using critical 10% of IoT device memory
- Supports both physical deployments and simulations (NS3)
- Real-time anomaly detection with cryptographic hashing for security

## Installation
To set up and run ENVOT, follow the steps below:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/ENVOT.git
    cd ENVOT
    ```

2. Install the required dependencies (assuming you have Python and pip installed):
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the NS3 simulation environment if you're running the large-scale simulations:
    - Follow the NS3 installation guide [here](https://www.nsnam.org/wiki/Installation)

4. (Optional) Configure hardware if you're running it on IoT devices like Arduino.

## Usage
### Running the Model on IoT Data
1. Place your dataset files in the `data/` directory.
2. Run the preprocessing script to prepare the data:
    ```bash
    python preprocess.py --input data/ --output processed_data/
    ```

3. Train the VAE model:
    ```bash
    python train_vae.py --data processed_data/ --output vae_model/
    ```

4. Run the anomaly detection:
    ```bash
    python anomaly_detection.py --model vae_model/ --data processed_data/
    ```

### Running Simulations
To run the NS3-based large-scale simulations:
```bash
ns3 run simulation_script.cc
```
