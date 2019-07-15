import rl
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from environment import WristCom, WristEnv

wrist_com = WristCom(portTX=30005, portRX=30004, udpIP="127.0.0.2")

class WristDQNAgent:

    def __init__():

# rot_env = WristEnv("rotation", wrist_com.robotObj[0].pmin, wrist_com.robotObj[0].pmax, wrist_com)

