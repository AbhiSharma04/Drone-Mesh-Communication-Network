# Team 1: Drone Mesh Communication Network

## Problem Statement

<p style='text-align: justify;'>
Every year, numerous forest fires wreak havoc, consuming millions of acres of land globally. These fires pose a significant threat to natural habitats, surpassing other forms of land destruction, and causing widespread devastation to homes and businesses. The United States Forest Service underscores that effective prevention of forest fire destruction lies in proactive land management and early detection strategies.
  
A considerable challenge arises from the fact that many forest fires initiate in remote, hard-to-monitor areas.<br> Often, these fires can propagate for weeks before being identified, making containment and extinguishment efforts considerably more challenging. The highest likelihood of forest fires occurs during hot and dry weather conditions. To enhance the capabilities of wildfire firefighters, there is a critical need for technology facilitating the management of wilderness areas and the early detection of forest fires when they are more manageable in their nascent stages.

Given the exponential nature of fire spread, the swiftness with which firefighters can identify these fires profoundly impacts their ability to control and extinguish them.<br> A promising solution to this challenge involves the implementation of a system comprising autonomously flying drones. These drones would traverse wilderness areas, utilizing advanced sensors and imagery to promptly detect the presence of forest fires. This proactive approach holds the potential to significantly improve the effectiveness of firefighting efforts in curbing the destructive impact of these fires.
</p>

## Introduction

<p style='text-align: justify;'>
To develop an effective system of drones, numerous design requirements must be meticulously considered to optimize the technology's performance. A paramount concern and top priority in this design are the communication protocols facilitating data exchange between the drones and a central HUB responsible for data collection and system control. Given that many forest fires occur in remote areas devoid of conventional communication infrastructure, ensuring reliable data communication becomes a critical challenge.<br>
  
To address this challenge, the most viable solution involves establishing a mesh communication network, leveraging the collaborative efforts of multiple drones to relay data across expansive areas back to a central HUB for seamless data acquisition and system control. The rationale behind this approach is grounded in the fact that flying drones at altitudes just above the tree line is essential for capturing optimal data from the targeted areas. However, this proximity to the ground makes direct data communication over long distances nearly unattainable.<br>

Therefore, the creation of a mesh communication network among a fleet of drones emerges as a crucial aspect of this technology's success. This network extension is imperative to ensure the design's viability and incorporates self-healing mechanisms to address potential issues or delays in data transmission or the flight programming of individual drones. By establishing a resilient mesh communication network, the technology can overcome the challenges posed by remote locations and guarantee the seamless functionality of the wildfire detection system.
</p>

## Project Design 

### Mesh Network and Extending Communication Range:

<p style='text-align: justify;'>
We are using zigbees to enable communication between drones and hub.These devices act as radios that can transmit and receive signals. All the drones and hub are equipped with a zigbee module. Each zigbee has a communication range of 100m in closed spaces and a range of 300m in open spaces. To extend the range and enhance the reliability of the communication network in the drone system, several strategic measures are implemented. <br>
  
First, a Range Extension approach is adopted by positioning nodes strategically beyond the direct reach of each other. This ensures that the communication coverage extends over a larger area, especially in remote or challenging terrains. <br>

Additionally, an Overlap Strategy is employed, ensuring that communication nodes have overlapping coverage areas. This guarantees continuous and seamless connectivity, minimizing potential communication gaps. To address long-distance data transfer challenges, the system utilizes Indirect Communication, employing intermediate nodes to relay messages over extended distances. <br>

Furthermore, a Data Hopping technique is implemented, allowing messages to jump across multiple nodes to reach their destination efficiently. Collectively, these strategies contribute to the robustness and efficiency of the communication network, enabling effective data transfer and coordination in the monitoring and control of the drone fleet. To achieve this all the signees should be in the same network with same configurations.</p>

### Drones
Each drone is integrated with a sensor module which consists of temperature, humidity, and wind speed sensors to collect values in a specific area. Every drone in the fleet is equipped with the same sensor systems, enhancing the system's versatility and enabling more comprehensive land coverage for fire detection. The design utilizes an Arduino to gather sensor signals, which are then processed and converted into digital hexadecimal code. This code is relayed through the mesh communication network back to the HUB.

Use this code for Drones: [`dronetohub.py`](drone/dronetohub.py)



### Hub:
<p style='text-align: justify;'>
In addition to the drones and sensor systems, the implementation of a central HUB is crucial for the successful execution of this design. The HUB serves as a central command and control center, endowed with the necessary capabilities to monitor and manage the entire system effectively.<br>
For monitoring purposes, the HUB is equipped with a Zigbee radio and functions as an integral part of the wireless mesh communication system, serving as the coordinator for the routed data transmitted from the sensors on the drones. Upon receiving this data, the HUB possesses the capability to store it for scientific analysis and machine learning applications. Machine learning algorithms can be integrated into the data acquisition system to discern patterns within the data measurements, enabling the system to make predictions and provide indications of potential fire existence based on identified patterns.<br>
In terms of system control, the HUB has the capacity to communicate with the flight controllers of all drones. It can relay GPS coordinates and flight directions to each drone, facilitating centralized control over the entire fleet. To achieve this, a Raspberry Pi is employed as a computing unit to govern the flight programming of the drones. This control system ensures the most efficient and optimal positioning of the drones to establish and maintain a mesh communication network over a designated area of interest.</p>

Use this code for Hub: [`recieving&sending_data_ .py`](Hub/dronetohub.py)


# Steps:

## Connecting Arduino with Sensors
### Requirements:
1.Arduino board
2.Sensors: Temperature, Humidity, and Wind Speed
3.Jumper wires
#### Steps:
1.Connect Each Sensor to the Arduino:
2.Use jumper wires to connect each sensor to the appropriate pins on the Arduino.
3.The specific connections depend on the type of sensors you are using. Usually, you will connect the power (VCC and GND) and data output pins of the sensors to the corresponding pins on the Arduino.
## Load Sensor Reading Program:
1.Upload a program to the Arduino to read data from the sensors. Use this code for Arduino: 
Use this code for Hub: [`Sensor_Data.ino`](drone/Sensor_Data.ino)

2.The program should be  reading data from each sensor and sending this data via a serial connection.

#### Connecting Arduino to Raspberry Pi
### Requirements:
1.Raspberry Pi (with Raspbian OS installed)
2.Arduino board
3.USB cable

## Steps:
### Prepare Raspberry Pi:

1.Ensure the Raspberry Pi is set up with Raspbian OS and has the necessary software tools. 2.You might need Python with serial communication libraries (pyserial).

### Connect Arduino to Raspberry Pi:

1.Use the USB cable to connect the Arduino to the Raspberry Pi.
2.The Raspberry Pi should recognize the Arduino as a serial device.
### Upload a Python Script on Raspberry Pi:
1.Upload the script in Python that uses the pyserial library to open the serial port and read the data sent from the Arduino.
2.The script should be able to parse the sensor data for further use or processing.

### Connecting XBee to Raspberry Pi
### Requirements:
1.Raspberry Pi
2.XBee module
3.XBee USB adapter or XBee Shield for Raspberry Pi

### Steps:
#### Connect XBee to Raspberry Pi:
1.If using an XBee USB adapter, connect the XBee to the adapter and plug it into a USB port on the Raspberry Pi.
2.If using an XBee Shield, mount the shield onto the GPIO pins of the Raspberry Pi and insert the XBee module into the shield.

## Configure XBee Module:

1.You have to configure the XBee module using software like XCTU .
2.Set up network parameters, like PAN ID and channel, to match other XBee modules in your network.
### Integrate XBee Communication in Raspberry Pi Script:

In our setup, the Raspberry Pi plays a crucial role in not only receiving data from the Arduino but also in sending this data through the XBee module for effective communication within the drone mesh network. This section of the documentation explains how this integration is achieved with the existing Python script.

### Overview of the Script's Functionality

Data Reception: The script on the Raspberry Pi is designed to read incoming data from the Arduino, which collects environmental information like temperature, humidity, and wind speed.

XBee Communication: The same script also handles the transmission of this data to the XBee module. This allows for the seamless relay of information within the network.


### Machine Learning:
<p style='text-align: justify;'>
In the proposed wildfire detection system, a machine learning algorithm is integrated to predict the occurrence of forest fires. A TensorFlow model is deployed on the central HUB, acting as the core computational unit for processing and analysis. This model is trained to interpret and learn from the sensor data collected by the fleet of drones. The sensors on each drone capture crucial environmental measurements such as temperature, humidity, and wind speed, along with additional data from photographs and infrared sensors.<br>
The TensorFlow model reads and analyzes this diverse set of sensor data to discern patterns and correlations indicative of potential fire incidents. Through the training phase, the model learns from historical data, understanding the complex relationships between various environmental factors and the presence of forest fires. Once deployed, the model can make real-time predictions based on incoming sensor data.</p>

### Dataset and Model Description:
<p style='text-align: justify;'>
We used a dataset of Algerian forest fires to train the model. The dataset consists of the following data. It includes columns for values of Temperature, Relative humidity, wind speed and the presence of fire on different days. </p>

### The script performs the following tasks:
1. Connects to Google Drive using Google Colab.
2. Imports required libraries for data analysis and machine learning.
3. Reads data from a CSV file containing forest fire data.
4. Performs data cleaning, handling missing values, and converting data types.
5. Explores and visualizes data using histograms and seaborn plots.
6. Splits the data into features (X) and target variable (y).
7. Identifies and removes highly correlated features.
8. Normalizes the data using the StandardScaler.
9. Trains a logistic regression model on the preprocessed data.
10. Evaluates the logistic regression model on test data.
11. Builds and trains a neural network model using TensorFlow.
12. Evaluates the neural network model on test data.
13. Saves the trained model in TensorFlow format.
14. Converts the saved TensorFlow model to TensorFlow Lite format.

#### Creating  a tensor flow model:

* Start by importing the necessary libraries, including TensorFlow, for building and training the machine learning model. 
* Load the dataset. Perform necessary preprocessing steps, such as handling missing values, converting categorical variables, and splitting the dataset into features (X) and labels (y).
* Split the dataset into training and testing sets to evaluate the model's performance.* Standardize the feature values to ensure that they are on a similar scale, improving model convergence.
* Construct a simple neural network using TensorFlow. The choice of model architecture depends on the complexity of the problem. 
* We used 3 different layers, one input layer with 64 nodes, one dense layer with 32 nodes and one output layer with one node to predict fire or no fire.
* We trained the model using the preprocessed training data and then assessed the model's performance on the test set to ensure generalization. 
* Now we use the trained model to make predictions on new data. The model has an accuracy of 91%.
* We developed a flask application and integrated the trained model to make predictions on incoming sensor data and display the result of prediction using a graphical user interface.

# Additional Resources

- **Code Documentation:** [View Code Documentation](Documentation/CodeDocumentation.md)
- **Setup Guide:** [View Setup Guide](Documentation/SetupGuide.md)
- **Project Report:** [View Report](Documentation/report_682.pdf)
