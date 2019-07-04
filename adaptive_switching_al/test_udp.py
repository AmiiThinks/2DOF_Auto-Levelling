import select, socket
from struct import *
import numpy as np
import time
import pickle

""" Initialization Section """
# Counter variables
number_of_steps = 0
start_time = None
numSwitchItems = 3


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

print("Starting main loop...")
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
                

        
            # Calculate the checksum        
            checksum = checksum_fcn(msgnew[2:len(msgnew)-1])
            packetRX.append(checksum)
            # Make sure the packet is the proper size, has the appropriate header, and the proper checksum
            if packetRX[0] == 255 and packetRX[1] == 255 and checksum == msgnew[len(msgnew)-1]:
                msg = msgnew
                UDP_flag = 1
                print(packetRX)
            else:
                failed_packets = failed_packets + 1
                print(failed_packets)

        # Start sending packets as soon as we make sure the connection is established    
        if UDP_flag == 1:
            
            number_of_steps += 1  # increase step count since beginning experiment



            # Construct the packet to transmit the predictions to the external script 
            N = numSwitchItems      # Number of servos for which data is being transmittted
            HEADER = 255            # Header byte
            LENGTH = 2*N            # The number of bytes in the data section of the packet
            DATA = []               # Load the data with the servoID and prediction for each servo
            
            for i in range(0, N):
                DATA.append(i + 1)          # add 1 to the SERVO ID to account for the servoID's starting at index of 1
            
            packetTX = [HEADER, HEADER, LENGTH]
            for i in DATA:   
                packetTX.append(i)
            checksumTX = checksum_fcn(packetTX[2:len(packetTX)])
            packetTX.append(checksumTX)
            packetTX_byte = bytearray(packetTX)     # convert the list to a byte array

            # Send the packet to the external script
            sock.sendto(packetTX_byte, (udpIP, portTX))
            UDP_flag = 0    # reset the flag, so we have to wait until the next packet is received from the external script


except KeyboardInterrupt:
    print("Quit")
    
# Close the communication socket                
sock.close()
