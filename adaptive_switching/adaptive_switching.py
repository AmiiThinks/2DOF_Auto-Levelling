import select, socket
from struct import *
import numpy as np
import time
from learning_toolkit import *
import pickle

""" Initialization Section """
# Prediction variables

alpha = 0.01        # The learning rate (higher means it learns faster, but may be unstable)
lambda_val = 0.99   # The lambda value in the TD lambda learner
gamma = 0.9         # Gamma is related to how far into the future we are predicting the joint activity (i.e. gamma of 0.9 means that we are anticipating/predicting into the future approximately 4/(1-gamma) = 40 steps under ideal conditions
rho = 1             # This is a parameter that relates to the eligibility traces used in the tile coder (1 = using eligibilty traces)
numTilings = 4      # The number of tilings in the tilecoder
numBins = 8       # The number of bins in the tilecoder

# Adaptive Switching Variables
numSwitchItems = 5      # Number of items in the switching list

# Initialize a separate TD Lambda Learner for each item in the switching list
#       - TDLambdaLearner(numTilings, num_bins (not used), alpha, lambda, gamma, cTableSize (not used))
td = [None]*numSwitchItems
td[0] = TDLambdaLearner(numTilings, 64, alpha, lambda_val, gamma, 64)  # TDLambda learner for gripper movement
td[1] = TDLambdaLearner(numTilings, 64, alpha, lambda_val, gamma, 64)  # TDLambda learner for gripper movement
td[2] = TDLambdaLearner(numTilings, 64, alpha, lambda_val, gamma, 64)  # TDLambda learner for gripper movement
td[3] = TDLambdaLearner(numTilings, 64, alpha, lambda_val, gamma, 64)  # TDLambda learner for gripper movement
td[4] = TDLambdaLearner(numTilings, 64, alpha, lambda_val, gamma, 64)  # TDLambda learner for gripper movement

# Counter variables
number_of_steps = 0
start_time = None

# Datalogging variables
list_step = []
list_shoulder_state = []
list_shoulder_pred = []
list_wristFlex_state = []
list_wristFlex_pred = []
list_hand_state = []
list_hand_pred = []

# On exiting/killing of the program write all the data to a pickle file for datalogging
def exit_handler():
    print('logging data...')
    experiment_data = open('test_data.txt', 'wb')
    pickle.dump([list_step, list_shoulder_state, list_shoulder_pred, list_wristFlex_state, list_wristFlex_pred, list_hand_state, list_hand_pred], experiment_data)
    experiment_data.close()
    print('data logged!')
    print('...exiting!')

# State Variables
class servo_info (object):
    # Abstract class to hold basic implementation for servo related information
    def __init__(self):
        self.position = 0   # The current position value of the servo
        self.velocity = 0   # The current moving speed of the servo
        self.load = 0       # The current load applied to the servo
        self.temp = 0       # The internal temperature of the servo in deg 
        self.state = 0      # The state of each motor (i.e. 0 = off, 1 = moving cw direction, 2 = moving in ccw direction, 3 = hanging until co-contraction is finished)
        
        self.normalized_position = 0
        self.normalized_velocity = 0
        self.normalized_load = 0
        self.normalized_temp = 0
        self.normalized_state = 0
        
    def update(self, position, velocity, load, temp, state):
        self.position = position
        self.velocity = velocity
        self.load = load
        self.temp = temp
        self.state = state

class mx_series(servo_info):
    # Specific instance of all servos of the mx series type (i.e mx-28, mx-64, mx-106)
        #* Position: 0 - 4095 (0.088 degrees/value, The central neutral position for each servo is 2048)
        #* Velocity: 0 - 2047 (0.11rpm/value, If the value reads 1024 then it means that the  servo is stationary. A value below 1023 means the motor rotates in a CCW direction and a value above 1023 means the motor rotates in a CW direction.)   
        #* Load: 0 - 2047 (0.1% load/value, If the value is 0 - 1023 it means the load works to the CCW direction and if the value is 1024-2047 it means the load works in the CW direction. A value of 1024 means that there is 0% load. )
        #* Temp: 0 - 255 (1 degC/value)
        #* State: 0 - 3 (see above for definitions of each state)
    def __init__(self):
        self.pmin = 0
        self.pmax = 0
        self.vmin = 0
        self.vmax = 0
        self.maxload = 0
        self.tempload = 0
        
    def update_full_norm(self, position, velocity, load, temp, state):
        servo_info.update(self, position, load, velocity, temp, state)
        # NOTE: These are the normalized values assuming the MX series servo is using the full range of its joint limits.
        #       i.e. if we are only using positions 0 to 4095 then this method will scale those values to be between 0 and 1.
        self.normalized_position = float(self.position/4095)
        self.normalized_velocity = float(self.velocity/2047)
        self.normalized_load = float(self.load/2047)
        self.normalized_temp = float(self.temp/255)
        if self.state == 1 or self.state == 2:
            self.normalized_state = 1   # moving cw or ccw
        else:
            self.normalized_state = 0   # not moving
            
    def update_zoomed_norm(self, position, velocity, load, temp, state):
        servo_info.update(self, position, velocity, load, temp, state)
        # NOTE: These are the normalized values assuming the MX series servo is using the zoomed in range of its joint limits for the Bento Arm. Each joint will in practise have different limits
        #       Zooming in on the specific range before feeding the information into the tile coder will help improve the resolution and scaling.
        #       i.e. if we are only using positions 1928 to 2800 on the chopsticks gripper then we should normalize so 1928 is 0 and 2800 is 1.
        self.normalized_position = always_positive(float((self.position - self.pmin)/(self.pmax - self.pmin)))
        self.normalized_velocity = always_positive(float((self.velocity - self.vmin)/(self.vmax - self.vmin)))
        self.normalized_load = always_positive(float((self.load - self.loadmin)/(self.loadmax - self.loadmin)))
        self.normalized_temp = always_positive(float(self.temp/self.maxtemp))
        if self.state == 1 or self.state == 2:
            self.normalized_state = 1   # moving cw or ccw
        else:
            self.normalized_state = 0   # not moving

# Helper function to make sure normalized values cannot be less than 0 which might mess up the tile coder
def always_positive(value):
    if value < 0:
        return 0
    else:
        return value

# Helper Function to make sure values being converted to bytes are in the range of 0 to 255
def byte_sized(value):
    if value > 255:
        return 255
    elif value < 0:
        return 0
    else:
        return value
        
# Create the list of servo objects and define their joint limits for optimal scaling into the tile coder
robotObj = []
buffer = 10  # add this value to joint limits to prevent normalized values from exceeding 1 or going below 0 if they slightly exceed their limits
robotObj.append(mx_series())
robotObj.append(mx_series())
robotObj.append(mx_series())
robotObj.append(mx_series())
robotObj.append(mx_series())
robotObj[0].pmin = 1664 - buffer
robotObj[0].pmax = 2432 + buffer
robotObj[0].vmin = 1024 - (55 + buffer)
robotObj[0].vmax = 1024 + (55 + buffer)
robotObj[0].loadmin = 1024 - (225 + buffer) 
robotObj[0].loadmax = 1024 + (225 + buffer)
robotObj[0].maxtemp = 80 + buffer
robotObj[1].pmin = 1784 - buffer
robotObj[1].pmax = 2570 + buffer
robotObj[1].vmin = 1024 - (45 + buffer) 
robotObj[1].vmax = 1024 + (45 + buffer)
robotObj[1].loadmin = 1024 - (300 + buffer) 
robotObj[1].loadmax = 1024 + (300 + buffer)
robotObj[1].maxtemp = 80 + buffer
robotObj[2].pmin = 1028 - buffer
robotObj[2].pmax = 3073 + buffer
robotObj[2].vmin = 1024 - (90 + buffer)
robotObj[2].vmax = 1024 + (90 + buffer)
robotObj[2].loadmin = 1024 - (250 + buffer)
robotObj[2].loadmax = 1024 + (250 + buffer)
robotObj[2].maxtemp = 80 + buffer
robotObj[3].pmin = 790 - buffer
robotObj[3].pmax = 3328 + buffer
robotObj[3].vmin = 1024 - (67 + buffer)
robotObj[3].vmax = 1024 + (67 + buffer)
robotObj[3].loadmin = 1024 - (300 + buffer) 
robotObj[3].loadmax = 1024 + (300 + buffer)
robotObj[3].maxtemp = 80 + buffer
robotObj[4].pmin = 1928 - buffer
robotObj[4].pmax = 2800 + buffer
robotObj[4].vmin = 1024 - (90 + buffer)
robotObj[4].vmax = 1024 + (90 + buffer)
robotObj[4].loadmin = 1024 - (400 + buffer) 
robotObj[4].loadmax = 1024 + (400 + buffer)
robotObj[4].maxtemp = 80 + buffer

# np.set_printoptions(formatter={'float_kind':'{:25f}'.format})
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

# Set the UDP ports and initialize the communication objects
portTX = 30003  # The port that this script will send data to
portRX = 30002  # The port that this script will receive data from
udpIP = "127.0.0.1" # The IP address of the computer/programt that you want to send data to. Use 127.0.0.1 when communicating between two programs on the same computer.
bufferSize = 1024  # buffer for incoming bytes
UDP_flag = 0 
failed_packets = 0

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((udpIP, portRX))
sock.setblocking(0)

# Calculate the checksum for the UDP packet
def checksum_fcn(packet):
    # The checksum is calculated via the following formula: ~ (LENGTH + FOR EACH SERVO[SERVO ID + PREDICTION], where ~ represents the NOT operation
    summed_packet = sum(packet)
    checksum = ~np.uint8(summed_packet)
    return checksum

""" Main Loop """
try:
    while True:
        #start = time.time()
        while True:
            # Check whether any data has been received from the external script
            result = select.select([sock], [], [], 0.0)
            if len(result[0]) == 0:
                break
            msgnew = result[0][0].recv(bufferSize)

            # Decode the packet that was defined in UDP_Comm_Protocol_ASD_brachIO_to_python_180619
            N = 9           # number of bytes per switching item
            DataNum = 6     # number of decoded items per switching item (this should be equal to smaller than N)
            index_counter = 0 # the index used to assign values to the robotObj 
            packetRX = []
            packetRX.append(msgnew[0])      # header byte
            packetRX.append(msgnew[1])      # header byte
            packetRX.append(msgnew[2])      # length byte
            for i in range(3,packetRX[2], N):
                packetRX.append(msgnew[i])  # Servo ID (use this code for single bytes)
                packetRX.append(int.from_bytes(msgnew[i+1:i+3], byteorder='little'))    # Servo Position (use this code for ushorts (low byte then high byte))
                packetRX.append(int.from_bytes(msgnew[i+3:i+5], byteorder='little'))    # Servo Velocity 
                packetRX.append(int.from_bytes(msgnew[i+5:i+7], byteorder='little'))    # Servo Load
                packetRX.append(msgnew[i+7])    # Servo temperature
                packetRX.append(msgnew[i+8])    # Servo state
                
                index_counter = len(packetRX) - DataNum     # Find the index that contains the switching item ID for the current item and use it as a reference for its related data values
                
                # Update the robot object with the decoded information and calculate the normalized values
                robotObj[packetRX[index_counter]-1].update_zoomed_norm(packetRX[index_counter+1],packetRX[index_counter+2],packetRX[index_counter+3],packetRX[index_counter+4],packetRX[index_counter+5])

            # Calculate the checksum        
            checksum = checksum_fcn(msgnew[2:len(msgnew)-1])
            packetRX.append(checksum)
            # Make sure the packet is the proper size, has the appropriate header, and the proper checksum
            if packetRX[0] == 255 and packetRX[1] == 255 and checksum == msgnew[len(msgnew)-1]:
                msg = msgnew
                UDP_flag = 1
                #print(packetRX)
            #else:
            #    failed_packets = failed_packets + 1
            #    print(failed_packets)

        # Start sending packets as soon as we make sure the connection is established    
        if UDP_flag == 1:
            
            number_of_steps += 1  # increase step count since beginning experiment

            # Update state information with the latest values received from the external program
            state_joints = [robotObj[0].normalized_position*numBins]
            #state_joints = [robotObj[0].normalized_position*numBins,
                        #robotObj[0].normalized_velocity*numBins]

            # Define the reward/cumulant for each TD learner
            # Since the state property has already been converted into a suitable cumulant this step is a bit redundant, but is included as a reference in case some sort of conversion might be required for other cumulant signals
            cumulant = [None]*numSwitchItems
            for i in range(numSwitchItems):
                cumulant[i] = robotObj[i].normalized_state
                
            # Update the TD learners - td.update(state input, gamma, reward)
            for i in range(numSwitchItems):
                td[i].update(state_joints, gamma, rho, cumulant[i])

            # Calculate the normalized predictions
            normalized_pred = [None]*numSwitchItems
            for i in range(numSwitchItems):
                normalized_pred[i] = td[i].prediction*(1-gamma)

            # Scale the normalized predictions into a range that fits nicely into a single byte
            scaled_pred = [None]*numSwitchItems
            for i in range(numSwitchItems):
                scaled_pred[i] = int(byte_sized(normalized_pred[i]*250))

            # Print the predictions
            #print(normalized_pred)
            print(scaled_pred)
            
            # Construct the packet to transmit the predictions to the external script 
            N = numSwitchItems      # Number of servos for which data is being transmittted
            HEADER = 255            # Header byte
            LENGTH = 2*N            # The number of bytes in the data section of the packet
            DATA = []               # Load the data with the servoID and prediction for each servo
            
            for i in range(0, N):
                DATA.append(i + 1)          # add 1 to the SERVO ID to account for the servoID's starting at index of 1
                DATA.append(scaled_pred[i])    # add the predictions for each switching item
            
            packetTX = [HEADER, HEADER, LENGTH]
            for i in DATA:   
                packetTX.append(i)
            checksumTX = checksum_fcn(packetTX[2:len(packetTX)])
            packetTX.append(checksumTX)
            packetTX_byte = bytearray(packetTX)     # convert the list to a byte array

            # Send the packet to the external script
            sock.sendto(packetTX_byte, (udpIP, portTX))
            UDP_flag = 0    # reset the flag, so we have to wait until the next packet is received from the external script

            # Data logging (should normally comment this out if you are not using it)
            list_step.append(number_of_steps)
            list_shoulder_state.append(robotObj[0].normalized_state)
            list_shoulder_pred.append(normalized_pred[0])
            list_wristFlex_state.append(robotObj[3].normalized_state)
            list_wristFlex_pred.append(normalized_pred[3])
            list_hand_state.append(robotObj[4].normalized_state)
            list_hand_pred.append(normalized_pred[4])

except KeyboardInterrupt:
    exit_handler()
    
# Close the communication socket                
sock.close()
