from flask import Flask, render_template, request, jsonify
import pandas as pd
import tensorflow as tf
import numpy as np
import time
import csv

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="/home/hub/Desktop/hub/converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)

def predict_fire(filepath):
    prediction = []

    # Read data from the CSV file
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        
        # Iterate over each row in the CSV file
        for row in reader:
            # Preprocess input data
            temperature = ((float(row['Temperature'])) - 32) / 1.8
            humidity = float(row['Humidity'])
            wind_speed = float(row['Wind Speed']) * 100
            preprocessed_input = np.array([[temperature, humidity, wind_speed]], dtype=np.float32)

            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], preprocessed_input)

            # Run inference
            interpreter.invoke()

            # Get the output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Process the output as needed
            threshold = 0.7
            prediction_binary = 1 if output_data[0, 0] > threshold else 0
            print("Model output:", prediction_binary)
            prediction.append(prediction_binary)
            
            # Optional: Add a delay to control the rate of inference
            # time.sleep(1)
    
    # Clear the CSV file after processing
    clear_csv(filepath)
    
    # Re-create the CSV file with headers
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Temperature', 'Humidity', 'Wind Speed'])
    
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_csv', methods=['GET'])
def predict_csv():
    # Retrieve the uploaded CSV file
    csv_file = "/home/hub/Desktop/hub/sensor_data.csv"

    # Check if a file is selected
    if not csv_file:
        return "No CSV file selected."
    
    # Perform prediction and render the template with predictions
    prediction = predict_fire(csv_file)
    return render_template('prediction_auto.html', predictions=prediction)

def clear_csv(filepath):
    # Clear the contents of the CSV file
    with open(filepath, 'w', newline='') as file:
        file.truncate()

if __name__ == '__main__':
    app.run(debug=True)
