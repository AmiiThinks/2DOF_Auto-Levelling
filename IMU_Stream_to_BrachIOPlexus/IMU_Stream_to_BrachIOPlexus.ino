// Analog Sensor Stream
// Version 1.0
// Author: Michael (Rory) Dawson
//
// Objective: This sketch streams analog input values to a PC over serial via USB or bluetooth
//
// Bluetooth hookup:
// TX-0 pin of bluetooth mate -> Arduino Pro Mini pin RX1 (GREEN)
// RX-I pin of bluetooth mate -> Arduino Pro Mini pin TX0 (YELLOW)
//
// References:
// Structs: http://playground.arduino.cc/Code/Struct
// Sample compiler for initial testing: http://ideone.com/IPWy58
// SerialEvent: https://www.arduino.cc/en/Tutorial/SerialEvent
// springf() -> adding padded zeros to strings: https://gist.github.com/philippbosch/5395696

//IMU Sensor Setup
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
  
Adafruit_BNO055 bno = Adafruit_BNO055(55);

// DEFINE number of sensor channels
const int ch_num = 4;

// Define timing parameters
const int timestep1 = 40; // timestep to update servo position and read values from FSRs in milliseconds
long timer1 = 0;
unsigned long currentMillis;

// for outgoing Serial data
char buff[4];   // Long enough to hold complete integer string
String feedback;
String concat_message;

// for reading in analog voltages
int current_value;

// Define global array of structs to hold the parameters for each sensor
struct sensorParam {
  int value;          // the analog input pin that will be assigned to this sensor channel 
  int enabled;             // whether the sensor is enabled or not 0 = false, 1 = true
} sensor[8]; // need to make one more than the amount used because of some kind of bug in the compiler

// This sketch outputs Serial data at 9600 baud (open Serial Monitor to view).

void setup()
{
  // Define sensor 0 parameters  
  sensor[1].value = 0;          // the analog input pin that will be assigned to this sensor channel                
  sensor[1].enabled = 1;              // whether the sensor is enabled or not 0 = false, 1 = true

  // Define sensor 1 parameters  
  sensor[2].value = 0;          // the analog input pin that will be assigned to this sensor channel                
  sensor[2].enabled = 1;              // whether the sensor is enabled or not 0 = false, 1 = true

  // Define sensor 2 parameters  
  sensor[3].value = 0;          // the analog input pin that will be assigned to this sensor channel                
  sensor[3].enabled = 1;              // whether the sensor is enabled or not 0 = false, 1 = true

  // Define sensor 2 parameters  
  sensor[4].value = 0;          // the analog input pin that will be assigned to this sensor channel                
  sensor[4].enabled = 1;              // whether the sensor is enabled or not 0 = false, 1 = true
  
  // set up Serial port for output
  // Use Serial for arduino leonardo/micro boards when using usb connection https://www.arduino.cc/en/Reference/Serial
  Serial.begin(9600);
//  while (!Serial) 
//  {
//    ; // wait for Serial port to connect. Needed for Leonardo only
//  }

//IMU sensor setup
  /* Initialise the sensor */
  if(!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }
  
  delay(1000);
    
  bno.setExtCrystalUse(true);

}

void loop() 
{
  // Check the number of milliseconds elapsed since the sketch started running
  currentMillis = millis();
  
  // Read in sensor data from IMU
  sensors_event_t event; 
  bno.getEvent(&event);
  imu::Vector<3> imu = bno.getVector(Adafruit_BNO055::VECTOR_GRAVITY);
  
//  // Map gravity values from 0 to 1023 for BrachI/Oplexus:
  imu.x() = max(min(map(imu.x()*1000,-9810,9810,0,1023),1023),0);
  imu.y() = max(min(map(imu.y()*1000,-9810,9810,0,1023),1023),0);
  imu.z() = max(min(map(imu.z()*1000,-9810,9810,0,1023),1023),0);
  
  // Send back voltage feedback from analog sensors every 40ms
  if (timer1 == 0)
  {
    timer1 = currentMillis;
  }
  else if (currentMillis >= (timer1 + timestep1))
  {
    concat_message = "";
    sensor[1].value = imu.x();
    sensor[2].value = imu.y();
    sensor[3].value = imu.z();   
    
    for(int m=1; m <= ch_num; m++)
    {  
      if (sensor[m].enabled == 1)
      {
        current_value = sensor[m].value;
        
        sprintf(buff,"%04d",current_value);
        feedback = buff;
        concat_message = concat_message + 'A' + feedback;
      }
    }
      timer1 = 0;
      Serial.println(concat_message);
  }
  
}
