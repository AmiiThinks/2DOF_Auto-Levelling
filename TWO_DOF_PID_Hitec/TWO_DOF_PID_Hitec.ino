//Set servo position to keep IMU position stable using PID control: rotation and flexion.
//Dylan Brenneis, March 12, 2018

//IMU Sensor Setup
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <Servo.h>
  
Adafruit_BNO055 bno = Adafruit_BNO055(55);

//Define PID Parameters
float Kp_ph = 0.03;       
float Ki_ph = 0.00;       
float Kd_ph = 0.08;         
float Kp_th = 0.03;       
float Ki_th = 0.0;       
float Kd_th = 0.05;         
float lastErr = 0;
float error = 0;
float setpoint_ph = 0;
float setpoint_th = 0;
float output_ph = 0;
float output_th = 0;
int lastTime = 0;
float errSum = 0;

//declare counter for IMU initialization
int counter = 0;

//Declare down vector variables
float g_mag = 0;
float a = 0;
float b = 0;
float c = 0;
float ph = 0;
float th = 0;

//Declare servo information
Servo S_Rot;
Servo S_Flx;
float S_Rot_pos = 80;
float S_Flx_pos = 110;

//%%%SETUP%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

void setup() {
  //Begin Serial Monitor
  Serial.begin(9600);
  
  //Initialize servos to starting positions
  S_Rot.attach(3);
  S_Flx.attach(5);
  S_Rot.write(S_Rot_pos);
  S_Flx.write(S_Flx_pos);
  
  //Initialize the IMU sensor
  if(!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }
  delay(1000);
    
  bno.setExtCrystalUse(true);
  
  //Read in the initial sensor position (to be used as setpoint).
  //Loop 500 times to allow the sensor time to settle.
  while(counter < 500)
  {
  Get_DV();
  counter++;
  }
  
  //Set setpoint targets to current position:  
  Get_DV();
  setpoint_ph = Get_ph(a,b,ph);
  setpoint_th = Get_th(b,c,th);

}

//%%%MAIN LOOP%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

void loop() {
  //Get current position
  Get_DV();
  ph = Get_ph(a,b,ph);
  th = Get_th(b,c,th);
  
  //Using current position and PID controller, find desired servo position and go there:
  Set_Position();
  
}

//%%%GET_DV%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

//Function to take in IMU readings and convert to normalized vector components:
void Get_DV()
{
  sensors_event_t event; 
  bno.getEvent(&event);
  imu::Vector<3> imu = bno.getVector(Adafruit_BNO055::VECTOR_GRAVITY);
  g_mag = magnitude(imu.x(),imu.y(),imu.z());
  a = normalize(imu.x(),g_mag);
  b = normalize(imu.y(),g_mag);
  c = normalize(imu.z(),g_mag);
}

//%%%SET_POSITION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

//Function to call PID controller, get the desired servo position, and write it to the servos:
void Set_Position()
{
  output_ph = PID_Controller(ph,setpoint_ph,Kp_ph,Ki_ph,Kd_ph);
  output_th = PID_Controller(th,setpoint_th,Kp_th,Ki_th,Kd_th);
  S_Rot_pos = max(min(S_Rot_pos -  output_ph,180),0);
  S_Flx_pos = max(min(S_Flx_pos - output_th,180),0);
  S_Rot.write(S_Rot_pos);
  S_Flx.write(S_Flx_pos); 
}

//%%%GET_PHI%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

//Function to find angle ph:
float Get_ph(float num, float den, float current_angle)
{
  //Conditions when den = 0:
  if(den==0){
    if(num<0){
    return 90.0;
    }
    else if(num>0){
    return 270.0;
    }
    else{
    return current_angle;
    }
  }
  
  //den greater than 0:
  else if(den>0){
    return atan(num/den)*180/3.14159 + 180;
  }
  //den less than 0 and num equal to or less than 0:
  else if(num<=0){
    return atan(num/den)*180/3.14159;
  }
  //den less than 0 and num greater than 0:
  else{
    return atan(num/den)*180/3.14159 + 360;
  }
}

//%%%GET_THETA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

//Function to find angle th:
float Get_th(float num, float den, float current_angle)
{
  //Conditions when den = 0:
  if(den==0){
    if(num<0){
    return 0.0;
    }
    else if(num>0){
    return 180.0;
    }
    else{
    return current_angle;
    }
  }
  
  //den greater than 0:
  else if(den>0){
    return atan(num/den)*180/3.14159 + 90;
  }
  //den less than 0 and num equal to or less than 0:
  else if(num<=0){
    return atan(num/den)*180/3.14159 + 270;
  }
  //den less than 0 and num greater than 0:
  else{
    return atan(num/den)*180/3.14159 + 270;
  }
}

//%%%PID_CONTROLLER%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

//PID Controller Function
float PID_Controller(float measured_value, float setpoint, float Kp, float Ki, float Kd)
{
  //how long since we last calculated?
  int now = millis();
  float timeChange = (float)(now - lastTime);
  
  //compute working variables:
  error = setpoint - measured_value;
  errSum /= timeChange;
  errSum += (error * timeChange);
  float dErr = (error - lastErr) / timeChange;
  
  
  //Compute PID Output
  return Kp * error + Ki * errSum + Kd * dErr;
    
  //remember some variables for next time
  lastErr = error;
  lastTime = now;
}

//%%%MAGNITUDE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

//Function to find the magnitude of the gravity vector:
float magnitude(float x_grav, float y_grav, float z_grav) 
{
return sqrt( sq(x_grav) + sq(y_grav) + sq(z_grav));
}

//%%%NORMALIZE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

//Function to normalize vector components:
float normalize(float component, float g_mag)
{
 return component/g_mag; 
}

