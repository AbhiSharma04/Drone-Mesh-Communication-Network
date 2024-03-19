/* 
  This code is intended to test the wind, temperature, and humidity sensors that last year's team
  used for their project. Based on the sample code provided by the sensor manufacturer.
  https://github.com/moderndevice/Wind_Sensor/blob/master/WindSensor/WindSensor.ino
*/

#include "Adafruit_Sensor.h"
#include "Adafruit_AM2320.h"
// Modern Device wind + temp sensor setup
#define windPin 7 // Analog pin used for wind data
#define tempPin 6 // Analog pin used for temperature data
// The AM2320 sensor interfaces thru I2C and therefore must use the SDA pin (A4) and SCL pin (A5)
Adafruit_AM2320 am2320 = Adafruit_AM2320(); 


// Adjust the zeroWindAdjustment until your sensor reads about zero with a glass over it
const float zeroWindAdjustment =  .23; // negative numbers yield smaller wind speeds and vice versa.

int TMP_Therm_ADunits;  //temp termistor value from wind sensor
float RV_Wind_ADunits;    //RV output from wind sensor 
float RV_Wind_Volts;
unsigned long lastMillis;
int TempCtimes100;
float zeroWind_ADunits;
float zeroWind_volts;
float WindSpeed_MPH;

void setup() {
  
  Serial.begin(57600); // To monitor data in the serial monitor, great for testing
  Serial.println("start"); // Good way to make sure serial monitor is set up properly
  
  am2320.begin();
  pinMode(A7, INPUT);  // Wind
  pinMode(A6, INPUT); // Temperature
  digitalWrite(A6, LOW); // Turn off pullups


}

void loop() {
  // Read the raw data from the sensor
  TMP_Therm_ADunits = analogRead(tempPin);
  RV_Wind_ADunits = analogRead(windPin);

  // Calculations from the github
  RV_Wind_Volts = (RV_Wind_ADunits *  0.0048828125);
  TempCtimes100 = (0.005 *((float)TMP_Therm_ADunits * (float)TMP_Therm_ADunits)) - (16.862 * (float)TMP_Therm_ADunits) + 9075.4;  
  zeroWind_ADunits = -0.0006*((float)TMP_Therm_ADunits * (float)TMP_Therm_ADunits) + 1.0727 * (float)TMP_Therm_ADunits + 47.172; 
  zeroWind_volts = (zeroWind_ADunits * 0.0048828125) - zeroWindAdjustment;  
  WindSpeed_MPH =  pow(((RV_Wind_Volts - zeroWind_volts) /.2300) , 2.7265);

  // Print to serial moniter
  Serial.print("Wind Speed (mph): ");
  Serial.println((float)WindSpeed_MPH);
  Serial.print("Temperature (F): "); 
  Serial.println((am2320.readTemperature()*9/5)+32);
  Serial.print("Humidity (% RH): "); 
  Serial.println(am2320.readHumidity());
  
  delay(2500);

}
