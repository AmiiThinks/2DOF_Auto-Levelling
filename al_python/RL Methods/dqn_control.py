import select, socket
import numpy as np
import time
from keras import models, layers, optimizers
from random import sample, randint, uniform
import sys

""" Initialization Section """
# Prediction variables

# On exiting/killing of the program write all the data to a pickle file for datalogging
def exit_handler():
    print('logging data...')
    # experiment_data = open('test_data.txt', 'wb')
    # pickle.dump([list_step, list_shoulder_state, list_shoulder_pred, list_wristFlex_state, list_wristFlex_pred, list_hand_state, list_hand_pred], experiment_data)
    # pickle.dump([list_step, list_wristRot_state, list_wristRot_pred, list_wristFlex_state, list_wristFlex_pred, list_hand_state, list_hand_pred], experiment_data)

    # experiment_data.close()
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
        self.loadmin = 0
        self.loadmax = 0
        self.maxtemp = 0
        self.autolevelling = 0
        
    def update_full_norm(self, position, velocity, load, temp, state, autolevelling):
        self.autolevelling = autolevelling
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
            
    def update_zoomed_norm(self, position, velocity, load, temp, state, autolevelling):
        self.autolevelling = autolevelling
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
    
    def print_params(self):
        return str(self.position) + " " + str(self.velocity) + " " + str(self.load) + " " + str(self.temp) + " " + str(self.state) + " " + str(self.autolevelling)

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

#Constants for joint limits and velocities
#Pmin and Pmax values from brachIO      
wristRot_pmin = 1028
wristRot_pmax = 3073
wristFlex_pmin = 790
wristFlex_pmax = 3328
wristRot_v = 125
wristFlex_v = 125


# Create the list of servo objects and define their joint limits for optimal scaling into the tile coder
robotObj = []
buffer = 10  # add this value to joint limits to prevent normalized values from exceeding 1 or going below 0 if they slightly exceed their limits
robotObj.append(mx_series())
robotObj.append(mx_series())

robotObj[0].pmin = wristRot_pmin - buffer
robotObj[0].pmax = wristRot_pmax + buffer
robotObj[0].vmin = 1024 - (wristRot_v + buffer)
robotObj[0].vmax = 1024 + (wristRot_v + buffer)
robotObj[0].loadmin = 1024 - (250 + buffer)
robotObj[0].loadmax = 1024 + (250 + buffer)
robotObj[0].maxtemp = 80 + buffer
robotObj[1].pmin = wristFlex_pmin - buffer
robotObj[1].pmax = wristFlex_pmax + buffer
robotObj[1].vmin = 1024 - (wristFlex_v + buffer)
robotObj[1].vmax = 1024 + (wristFlex_v + buffer)
robotObj[1].loadmin = 1024 - (300 + buffer) 
robotObj[1].loadmax = 1024 + (300 + buffer)
robotObj[1].maxtemp = 80 + buffer


# np.set_printoptions(formatter={'float_kind':'{:25f}'.format})
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

# Set the UDP ports and initialize the communication objects
portTX = 30005  # The port that this script will send data to
portRX = 30004  # The port that this script will receive data from
udpIP = "127.0.0.2" # The IP address of the computer/programt that you want to send data to. Use 127.0.0.1 when communicating between two programs on the same computer.
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
    
def check_moving_by_vel(servos_to_check, t_l, t_h):
    for i in servos_to_check:
        if robotObj[i-1].normalized_velocity < t_l or robotObj[i-1].normalized_velocity > t_h:
            return True
    return False

def normalize(value, min, max):
    return (value - min)/(max - min)

def deg_to_ticks(degrees):
    return int((degrees * 11.3611111111111111111111)); #11.361111 degrees per encoder tick


""" RL Variables """

class replay_buffer:
    def __init__(self, buffer_size, state_size):
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.buffer = np.empty(shape=(self.buffer_size, self.state_size+1+1+self.state_size))

        self.buffer_count = 0
        self.buf_idx = 0

    def sample(self, batch_size):
        sampled_buf = self.buffer[np.random.choice(self.buffer.shape[0], size=batch_size, replace=False), :]
        return sampled_buf[:, 0:self.state_size], sampled_buf[:, self.state_size], sampled_buf[:, self.state_size+1], sampled_buf[:, self.state_size+2:self.state_size*2 + 2]

    def add_row(self, new_row):
        self.buffer[self.buf_idx, :] = new_row
        self.buf_idx = self.buf_idx + 1
        if self.buf_idx == self.buffer_size:
            self.buf_idx = 0
        if self.buffer_count < self.buffer_size:
            self.buffer_count = self.buffer_count + 1

    def is_full(self):
        if self.buffer_count == self.buffer_size:
            return True
        else:
            return False

    def clear(self):
        self.buffer_count = 0
        self.buf_idx = 0



class agent:

    def __init__(self, state_size, action_size, hidden_layers, activations, buffer_size, batch_size, epsilon, gamma, learning_rate, steps_to_update_target):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.prev_state = None
        self.prev_action = None
        self.epsilon = epsilon
        self.gamma = gamma

        self.model = models.Sequential()
        self.model.add(layers.Dense(hidden_layers[0], activation=activations[0], input_shape=(self.state_size, )))
        for i in range(1, len(hidden_layers)):
            self.model.add(layers.Dense(hidden_layers[i], activation=activations[i]))
        self.model.add(layers.Dense(self.action_size))
        self.model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss="mean_squared_error")

        self.update_target_model()

        self.steps_to_update_target = steps_to_update_target
        self.current_steps = 0

        self.replay_buf = replay_buffer(buffer_size, self.state_size)
    
    def train_on_buffer(self):
        if self.replay_buf.is_full():
            prev_state_buf, prev_action_buf, reward_buf, new_state_buf = self.replay_buf.sample(self.batch_size)

            prev_Q = self.model.predict(prev_state_buf, batch_size=self.batch_size)
            new_Q = self.target_model.predict(new_state_buf, batch_size=self.batch_size)
            target = reward_buf + self.gamma * np.amax(new_Q, axis=1)

            for i in range(self.batch_size):
                prev_Q[i, int(prev_action_buf[i])] = target[i]

            result = self.model.fit(prev_state_buf, prev_Q, batch_size=self.batch_size, epochs=1, verbose=0)

            self.current_steps = self.current_steps + 1
            if self.current_steps == self.steps_to_update_target:
                self.update_target_model()
                self.current_steps = 0

            return result.history['loss']
    
    def select_action(self, new_state):
        rand_num = uniform(0, 1)
        if rand_num < self.epsilon:
            action = randint(0, self.action_size-1)
        else:
            Q = self.model.predict(np.array( [new_state,]))
            action = np.argmax(Q)
        
        self.prev_state = new_state
        self.prev_action = action

        return action

    def add_to_buffer(self, reward, new_state):
        if self.prev_state is not None and self.prev_action is not None:
            self.replay_buf.add_row(self.prev_state + [self.prev_action] + [reward] + new_state)
            return True
        else:
            return False

    def reset(self):
        self.prev_state=None
        self.prev_action=None

    def update_target_model(self):
        self.target_model = models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

rot_action_size = wristRot_pmax - wristRot_pmin
loss = None
state_size = 5
rot_agent = agent(state_size, rot_action_size, [rot_action_size//4, rot_action_size//2], ['relu', 'relu'], 100, 32, 0.05, 0.6, 0.01, 10)
actions = [None, None]
rot_agent.model.summary()
""" Main Loop """
print("Starting")
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
            N = 10           # number of bytes per switching item
            DataNum = 7     # number of decoded items per switching item (this should be equal to smaller than N)
            index_counter = 0 # the index used to assign values to the robotObj 
            packetRX = []
            packetRX.append(msgnew[0])      # header byte
            packetRX.append(msgnew[1])      # header byte
            packetRX.append(msgnew[2])      # length byte

            for i in [3, 13, 23]:
                if (msgnew[i] != 2):
                    packetRX.append(msgnew[i])  # Servo ID (use this code for single bytes)
                    packetRX.append(int.from_bytes(msgnew[i+1:i+3], byteorder='little'))    # Servo Position (use this code for ushorts (low byte then high byte))
                    packetRX.append(int.from_bytes(msgnew[i+3:i+5], byteorder='little'))    # Servo Velocity 
                    packetRX.append(int.from_bytes(msgnew[i+5:i+7], byteorder='little'))    # Servo Load
                    packetRX.append(msgnew[i+7])    # Servo temperature
                    packetRX.append(msgnew[i+8])    # Servo state
                    packetRX.append(msgnew[i+9])    # Autolevelling on/off
                    index_counter = len(packetRX) - DataNum     # Find the index that contains the switching item ID for the current item and use it as a reference for its related data values
                    # Update the robot object with the decoded information and calculate the normalized values
                    robotObj[packetRX[index_counter]].update_zoomed_norm(packetRX[index_counter+1],packetRX[index_counter+2],packetRX[index_counter+3],packetRX[index_counter+4],packetRX[index_counter+5],packetRX[index_counter+6])
                else:
                    x_component = int.from_bytes(msgnew[i+1:i+3], byteorder='little', signed=True)    # x component of IMU
                    y_component = int.from_bytes(msgnew[i+3:i+5], byteorder='little', signed=True)   # adjusted y component of IMU
                    z_component = int.from_bytes(msgnew[i+5:i+7], byteorder='little', signed=True)   # z component of IMU
                    phi = int.from_bytes(msgnew[i+7:i+9], byteorder='little', signed=True)
                    setpoint_phi = int.from_bytes(msgnew[i+9:i+11], byteorder='little', signed=True)
                    theta = int.from_bytes(msgnew[i+11:i+13], byteorder='little', signed=True)
                    setpoint_theta = int.from_bytes(msgnew[i+13:i+15], byteorder='little', signed=True)
                    resetEnv = msgnew[i+15]
                    

            # Calculate the checksum        
            checksum = checksum_fcn(msgnew[2:len(msgnew)-1])
            packetRX.append(checksum)
            # Make sure the packet is the proper size, has the appropriate header, and the proper checksum
            if packetRX[0] == 255 and packetRX[1] == 255 and checksum == msgnew[len(msgnew)-1]:
                UDP_flag = 1
                #print(packetRX)
            #else:
            #    failed_packets = failed_packets + 1
            #    print(failed_packets)

        # Start sending packets as soon as we make sure the connection is established    
        if UDP_flag == 1:
            if resetEnv == 1:
                print("Reset")
                actions = [wristRot_pmax - wristRot_pmin, 0]
            else:
            #     new_state = [normalize(phi, 0, 360), normalize(setpoint_phi, 0, 360), robotObj[0].normalized_position, normalize(theta, 0, 360), normalize(setpoint_theta, 0, 360), robotObj[1].normalized_position,
            #     normalize(x_component, -512, 512), normalize(y_component, -512, 512), normalize(z_component, -512, 512)]

                new_state = [normalize(phi, 0, 360), normalize(setpoint_phi, 0, 360), robotObj[0].normalized_position, normalize(x_component, -512, 512), normalize(y_component, -512, 512)]
                reward = - deg_to_ticks(abs(setpoint_phi - phi))
                if rot_agent.add_to_buffer(reward, new_state):
                    loss = rot_agent.train_on_buffer()

                actions[0] = int(rot_agent.select_action(new_state)) + wristRot_pmin # - (rot_action_size)//2
                actions[1] = 0

                print("Actions: {0}, {1} | Rewards: {2} | Loss: {3} ".format(actions[0], actions[1], reward, loss))

            # Construct the packet to transmit the predictions to the external script 
            HEADER = 255            # Header byte
            LENGTH = 4              # The number of bytes in the data section of the packet
            DATA = []               # Load the data with the servoID and prediction for each servo
            for i in range(2):
                actionBytes = actions[i].to_bytes(2, 'little', signed=True)
                DATA.append(actionBytes[0])    # lower byte
                DATA.append(actionBytes[1])    # upper byte

            packetTX = [HEADER, HEADER, LENGTH]
            for i in DATA:   
                packetTX.append(i)
            # print(DATA)
            checksumTX = checksum_fcn(packetTX[2:len(packetTX)])
            packetTX.append(checksumTX)
            packetTX_byte = bytearray(packetTX)     # convert the list to a byte array

            # Send the packet to the external script
            sock.sendto(packetTX_byte, (udpIP, portTX))
            UDP_flag = 0    # reset the flag, so we have to wait until the next packet is received from the external script

except KeyboardInterrupt:
    exit_handler()
    
# Close the communication socket                
sock.close()
