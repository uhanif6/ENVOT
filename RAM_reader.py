import serial
import time
import os

# Define the serial port and baud rate
SERIAL_PORT = 'COM8'  # Replace with your serial port
BAUD_RATE = 9600
SAVE_PATH = 'E:\\Arduino\\RAM_dataset'  # Path to save the files

# Ensure the save directory exists
os.makedirs(SAVE_PATH, exist_ok=True)

def read_ram_from_arduino(serial_connection):
    try:
        line = serial_connection.readline().decode('utf-8').strip()
        if line.startswith("RAM Data:"):
            ram_data = line[len("RAM Data:"):].strip().split()
            return [int(x, 16) for x in ram_data]
    except Exception as e:
        print(f"Failed to read from Arduino: {e}")
    return []

def save_ram_data_to_file(ram_data, filename):
    with open(filename, 'w') as f:
        for byte in ram_data:
            f.write(f"{byte:02x} ")
        f.write("\n")

# Retry logic for opening the serial port
max_retries = 3
for attempt in range(max_retries):
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            time.sleep(2)  # Wait for the serial connection to initialize
            for i in range(10):  # Adjust the range for the number of reads you want
                ram_data = read_ram_from_arduino(ser)
                if ram_data:
                    filename = os.path.join(SAVE_PATH, f"DATA_file.txt")
                    save_ram_data_to_file(ram_data, filename)
                    print(f"Saved RAM data to {filename}")
                    break  # Exit the loop after one successful read
                time.sleep(2)  # Delay between reads
        break  # Exit the retry loop if successful
    except serial.SerialException as e:
        print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
        if attempt < max_retries - 1:
            print("Retrying...")
        else:
            print("All attempts to open the serial port failed.")