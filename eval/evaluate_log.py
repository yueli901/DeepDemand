import os
import re
import pandas as pd
import numpy as np

def extract_metrics(log_path):
    """
    Reads the log file and extracts training and validation metrics.
    """
    # Initialize empty lists to store data
    epochs = []
    sensors = []
    train_geh = []
    valid_geh = []

    # Regular expressions to capture relevant data from logs
    epoch_sensor_pattern = re.compile(r'Epoch (\d+), Sensor (\d+)')
    train_pattern = re.compile(
        r'Train Loss: MGEH: ([\d.]+)'
    )
    valid_pattern = re.compile(
        r'Valid Loss: MGEH: ([\d.]+)'
    )

    # Read the log file
    with open(log_path, 'r') as f:
        for line in f:
            epoch_sensor_match = epoch_sensor_pattern.search(line)
            train_match = train_pattern.search(line)
            valid_match = valid_pattern.search(line)

            if train_match:
                epoch = int(epoch_sensor_match.group(1))
                sensor = int(epoch_sensor_match.group(2))
                epochs.append(epoch)
                sensors.append(sensor)

                train_geh.append(float(train_match.group(1)))

            if valid_match:
                valid_geh.append(float(valid_match.group(1)))

    # Create a DataFrame
    df = pd.DataFrame({
        'epoch': epochs,
        'sensor': sensors,
        'train_geh': train_geh,
        'valid_geh': valid_geh,
    })


    return df