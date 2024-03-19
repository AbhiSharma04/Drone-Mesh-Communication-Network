from digi.xbee.devices import XBeeDevice, RemoteXBeeDevice, XBee64BitAddress
import time
import csv
import re
from datetime import datetime
import os
import random  # Added for simulated sensor data

# Dictionary to store the latest sensor data
latest_sensor_data = {"Temperature": None, "Humidity": None, "Wind Speed": None}

# Function to parse received sensor data and update the latest_sensor_data dictionary
def parse_and_update_sensor_data(data_str):
    try:
        if "Temperature" in data_str:
            temp_match = re.search(r"Temperature \(F\): ([\d.]+)", data_str)
            if temp_match:
                latest_sensor_data["Temperature"] = float(temp_match.group(1))
        elif "Humidity" in data_str:
            humidity_match = re.search(r"Humidity \(% RH\): ([\d.]+)", data_str)
            if humidity_match:
                latest_sensor_data["Humidity"] = float(humidity_match.group(1))
        elif "Wind Speed" in data_str:
            wind_speed_match = re.search(r"Wind Speed \(mph\): ([\d.]+)", data_str)
            if wind_speed_match:
                latest_sensor_data["Wind Speed"] = float(wind_speed_match.group(1))
    except Exception as e:
        print(f"Error parsing and updating sensor data: {e}")

# Function to check if all sensor data is collected (non-None)
def all_data_collected():
    return all(value is not None for value in latest_sensor_data.values())

# Function to save sensor data to a CSV file
def save_data_to_csv(filename='/home/hub/Desktop/hub/sensor_data.csv'):
    try:
        file_exists = os.path.isfile(filename)
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Timestamp', 'Temperature', 'Humidity', 'Wind Speed'])
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([current_time, latest_sensor_data["Temperature"], latest_sensor_data["Humidity"], latest_sensor_data["Wind Speed"]])
            for key in latest_sensor_data:
                latest_sensor_data[key] = None
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

# XBee device for communication with the remote XBee
remote_xbee = XBeeDevice("/dev/ttyUSB0", 115200)

# Function to send data to the remote XBee device
def send_data(data):
    try:
        # Specify the remote XBee's 64-bit address
        remote_device = RemoteXBeeDevice(remote_xbee, XBee64BitAddress.from_hex_string("13A20042010691"))
        remote_xbee.send_data_async(remote_device, data)
        print(f"Sent Data to Remote: {data}")
    except Exception as e:
        print(f"Error in sending data to remote: {e}")

# Simulated sensor data generator (for demonstration purposes)
def generate_simulated_sensor_data():
    return f"Temperature (F): {random.uniform(60, 80)}, Humidity (% RH): {random.uniform(30, 70)}, Wind Speed (mph): {random.uniform(0, 10)}"

try:
    # Open the remote XBee device for communication
    remote_xbee.open()
    print("Remote XBee Operational")

    while True:
        try:
            # Read data from the remote XBee module with a timeout of 2 seconds
            xbee_message = remote_xbee.read_data(2)
            if xbee_message is not None:
                # Decode and print the received data
                received_data = xbee_message.data.decode('utf-8')
                print(f"Received Data from Remote: {received_data}")
                # Parse and update the latest_sensor_data dictionary
                parse_and_update_sensor_data(received_data)
                # If all sensor data is collected, save it to the CSV file
                if all_data_collected():
                    save_data_to_csv()

            # Sending data (for demonstration, sending simulated sensor data)
            simulated_sensor_data = generate_simulated_sensor_data()
            send_data(simulated_sensor_data)

            # Sleep for 1 second before the next iteration
            time.sleep(1)

        except Exception as e:
            print(f"An error occurred: {e}")

except KeyboardInterrupt:
    pass
finally:
    try:
        # Close the remote XBee connection when done
        remote_xbee.close()
    except Exception as e:
        print(f"Error closing remote XBee connection: {e}")
