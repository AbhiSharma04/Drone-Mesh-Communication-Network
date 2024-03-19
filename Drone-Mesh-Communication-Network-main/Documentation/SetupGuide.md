# Project SetUp Guide

# Introduction
<p style='text-align: justify;'>
Annual global forest fires pose a severe threat to natural habitats and homes, necessitating proactive prevention through land management and early detection. Autonomous drones equipped with advanced sensors emerge as a promising solution, particularly for hard-to-monitor areas. Their swift deployment during the incipient stages of a fire, especially in hot and dry conditions, enhances firefighting effectiveness. Designing an effective drone system requires a focus on communication protocols, prioritizing the establishment of a reliable mesh network. This network extension, with self-healing mechanisms, ensures seamless functionality, enabling timely wildfire detection in challenging and remote terrains. You can follow the following steps to setup the project.
</p>

# Prerequisites

## Hardware for one drone and hub:

- 2 x RaspberryPi
- 3 x XBee
- 1 x Temperature and Humidity Sensor
- 1 x Wind Speed Sensor
- 1 x Arduino

## Software:

- XCTU 
- Arduino IDE
- Python Environment (with TensorFlow, flask Server and Digi Xbee modules installed)
- Real VNC
- Git

# Installation and SetUp:

## Hardware Setup:

1. Configuring XBee for Hub:
- Connect your XBee modules to your computer using an XBee Explorer USB or another compatible interface.
- Launch XCTU on your computer.
- Click on the "Discover" button in XCTU to identify the connected XBee modules. This will list all the XBee modules connected to your computer.
- Choose one XBee module to act as the coordinator. The coordinator is the main node that starts and manages the mesh network.
- For the coordinator, set the operating PAN ID (Personal Area Network ID). All nodes in the mesh network must have the same PAN ID to communicate with each other.
- XBee modules can operate as coordinators, routers, or end devices. Configure each module with the appropriate role using the "CE" (Coordinator Enable), "RE" (Router Enable), and "DE" (End Device) settings.

2. Configuring XBee for Drones:
- Connect your XBee modules to your computer using an XBee Explorer USB or another compatible interface.
- Launch XCTU on your computer.
- Click on the "Discover" button in XCTU to identify the connected XBee modules. This will list all the XBee modules connected to your computer.
- Choose one XBee module to act as the coordinator. The coordinator is the main node that starts and manages the mesh network.
- For other XBee modules (end devices or routers), set the same PAN ID as the coordinator. Also, configure the destination address (DH and DL) of these modules to match the coordinator's address.
- XBee modules can operate as coordinators, routers, or end devices. Configure each module with the appropriate role using the "CE" (Coordinator Enable), "RE" (Router Enable), and "DE" (End Device) settings.

3. Connecting RPi to Xbee:
- Connect the VCC pin of the XBee module to a 3.3V power source on the Raspberry Pi. Avoid connecting it to the 5V GPIO pin, as XBee modules typically operate at 3.3V.
- Connect the GND pin of the XBee module to a ground (GND) pin on the Raspberry Pi.
- Connect the TX (transmit) pin of the XBee module to the RX (receive) pin on the Raspberry Pi.
- Connect the RX (receive) pin of the XBee module to the TX (transmit) pin on the Raspberry Pi.
- Power up the Raspberry Pi, and the XBee module should start communicating with it.

4. Connect Temperature and Humidity Sensor to Arduino:
- The DHT sensors typically have three pins: VCC (power), GND (ground), and a signal/data pin.
- Connect the VCC pin of the DHT sensor to the 5V output on the Arduino.
- Connect the GND pin of the DHT sensor to one of the GND (ground) pins on the Arduino.
- Connect the data (signal) pin of the DHT sensor to a digital input/output pin on the Arduino. Choose a pin and remember the number (e.g., pin 2).

5. Connect WindSpeed Sensor to Arduino:
- The wind speed sensor typically has three pins: VCC (power), GND (ground), and a signal/data pin.
- Connect the VCC pin of the wind speed sensor to the 5V output on the Arduino.
- Connect the GND pin of the wind speed sensor to one of the GND (ground) pins on the Arduino.
- Connect the data (signal) pin of the wind speed sensor to a digital input pin on the Arduino. Choose a pin and remember the number (e.g., pin 2).

6. Connect Arduino to rpi (Drone):
- Plug one end of the USB cable into the USB port on the Arduino board.
- Connect the other end of the USB cable to one of the USB ports on the Raspberry Pi.
- If your Arduino is not externally powered, it will be powered through the USB connection from the Raspberry Pi.

## Software Setup:

1. Install python environment on rpi's of hub and drone
2. Install the Digi.Xbee modules
3. Install Tensorflow module on Hub - Rpi
4. Install XCTU Software to configure XBee's.
5. Configure Flask Web server on Hub - Rpi.
6. Install Arduino IDE.

# Running the Project:

1. Clone the Repository https://github.com/arkeerthana1/Drone-Mesh-Communication-Network
2. Ensure the hardware configurations are done properly.
3. Use XCTU to configure XBee's.
4. Upload the sketch sensordata.ino to arduino.
5. Connect the drone setup to a power supply to start collecting and transmitting sensor data to hub.
6. In the Hub-rpi, run the Receiving&Sending_data_.py file to receive the data sent from drones.
7. This data will be written to a csv file in hub.
8. Run the flask application in the terminal with the command - python3 firePredictionTemplate.py
9. This will start the webserver on localhost. copy the url in a browser.
10. click on predict to get the predictions from Machine Learning model using the data collected into csv file.

# Running Machine Learning Model:

1. Open the file ForestFirePredictionModel.ipynb in jupyter notebook.
2. Download the dataset into local machine and set the path accordingly in the jupyter notebook.
3. Run the file to create, build and save the Prediction model.


