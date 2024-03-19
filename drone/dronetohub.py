from digi.xbee.devices import XBeeDevice, RemoteXBeeDevice, XBee64BitAddress
from digi.xbee.exception import TimeoutException
import serial
import time

# Open a serial connection for the local device
ser = serial.Serial("/dev/ttyUSB0", 57600)

# Create an XBee device for communication with the remote device
xbee = XBeeDevice("/dev/ttyUSB1", 115200)

def read_serial_data():
    # Read data from the local serial port
    if ser.in_waiting > 0:
        try:
            # Decode and strip newline characters from the received data
            data = ser.readline().decode('utf-8').rstrip()
        except UnicodeDecodeError:
            print("Error decoding data")
        return data
    return None

def send_data(data):
    # Create a remote device object using the XBee address of the destination device
    remote_device = RemoteXBeeDevice(xbee, XBee64BitAddress.from_hex_string("13A200420107CE"))
    try:
        # Send data asynchronously to the remote device
        xbee.send_data_async(remote_device, data)
    except Exception as e:
        print(f"Error in sending data: {e}")

# Function to process received data
def process_received_data(received_data):
    # Add your custom processing logic here
    processed_data = received_data.upper()  # Example: Convert to uppercase
    return processed_data

# Function to log data to a file
def log_to_file(data, filename="log.txt"):
    with open(filename, "a") as file:
        file.write(f"{time.ctime()}: {data}\n")

try:
    # Open the XBee device for communication
    xbee.open()
    print("XBee Hub Operational")

    while True:
        # Sending data
        sensor_data = read_serial_data()
        if sensor_data:
            print("Sending data:", sensor_data)
            send_data(sensor_data)

            # Log the sent data to a file
            log_to_file(f"Sent: {sensor_data}")

        # Receiving data
        try:
            # Read data from the XBee module with a timeout of 1 second
            xbee_message = xbee.read_data(1)
            if xbee_message is not None:
                # Decode and print the received data
                received_data = xbee_message.data.decode('utf-8')
                print(f"Received Data: {received_data}")

                # Process received data
                processed_data = process_received_data(received_data)
                print(f"Processed Data: {processed_data}")

                # Log the received and processed data to a file
                log_to_file(f"Received: {received_data}, Processed: {processed_data}")

        except TimeoutException:
            print("No data received within the timeout period.")

except KeyboardInterrupt:
    pass
finally:
    # Close the XBee and serial connections when done
    xbee.close()
    ser.close()
