import math
import numpy as np
from simulation_tools import PID
from simulation_tools import PRBS
from scipy.integrate import odeint
from scipy import signal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Model


        
class Servo_2DOF:   
    def __init__(self):
        self.pid_rot = PID(1, 1, 1)
        self.pid_flex = PID(1, 1, 1)
    
    def neural_model_from_weights(self, weights):
        self.neural_model = Sequential()
        self.neural_model.add(LSTM(16, batch_input_shape=(1, None, 2), return_sequences=True, stateful=True))
        self.neural_model.add(Dense(2))
        self.neural_model.layers[0].set_weights(weights[0])
        self.neural_model.layers[1].set_weights(weights[1])
        
    def simulate_step_model_free(self, y0, cur_angle, t, r, d):
        #y0 should be [rot_pos, flex_pos]
        e = np.zeros(2)
        u = np.zeros(2)

        e = r - cur_angle
        
        u[0] = self.pid_rot.update(e[0], t)
        u[1] = self.pid_rot.update(e[1], t)

        u = u * math.pi/180 + y0
        
        model_input = np.array(u).reshape(1, 1, 2)
        model_output = self.neural_model.predict(model_input, batch_size=1)
        y = model_output[0] #First time step (our sequence length is one anyways)
        next_angle = y * 180/math.pi + 180 + d

        return y, next_angle
    
    def reset(self):
        self.pid_rot.reset()
        self.pid_flex.reset()
        self.neural_model.reset_states()

        
def simulation_init_2dof(time_step, length, aprbs_hold, aprbs_amp):
    T = np.arange(0, length, time_step)
    r = np.zeros(shape=(len(T), 2))
    r[:] = 180
    
    sig_gen = PRBS()
    d = np.zeros(shape=(len(T), 2))
    d[:, 0] = sig_gen.apply_butter(sig_gen.generate_APRBS(len(T), aprbs_hold, aprbs_amp))
    d[:, 1] = sig_gen.apply_butter(sig_gen.generate_APRBS(len(T), aprbs_hold, aprbs_amp))
    
    d_test = np.zeros(shape=(len(T), 2))
    d_test[:] = 0

    quarter = len(T)//4

    d_test[quarter:quarter*2, 0] = 45
    d_test[quarter*2:quarter*3, 1] = 45

    y0 = [0, 0]
    
    return T, r, d, d_test, y0
