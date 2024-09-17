import os
import subprocess

def run_simulation():
    # Run the NS-3 simulation
    subprocess.run(["./waf", "--run", "scratch/iot_simulation"], check=True)

def run_ml_model():
    # Run the Python model script
    subprocess.run(["python3", "scratch/ml_model.py"], check=True)

def main():
    # Step 1: Run NS-3 simulation
    run_simulation()
    
    # Step 2: Process the output data with the ML model
    run_ml_model()

if __name__ == "__main__":
    main()
