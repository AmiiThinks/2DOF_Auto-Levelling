#Samples angle only every 40ms
#Added validation set
#Changing init
#Added leakyRelu

#Python file for running in the background

# In[2]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.models import Model
from keras.initializers import TruncatedNormal, Ones, Zeros
import time
import pickle as pkl


# In[3]:


from simulation_tools import Servo, PRBS, simulation_init


# In[4]:


class NeuralNetwork:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(4, input_shape=(4,), 
                            kernel_initializer=TruncatedNormal(mean=0, stddev=0.1),
                            bias_initializer=TruncatedNormal(mean=0, stddev=0.1)))
        self.model.add(LeakyReLU(alpha=0.3, name="intermediate_layer"))
        self.model.add(Dense(3,
                            kernel_initializer=TruncatedNormal(mean=0, stddev=0.1),
                            bias_initializer=TruncatedNormal(mean=0, stddev=0.1)))
        
        self.stored_weights = None
        self.stored_weight_layer = None
        self.update_intermediate_model()

    
    def predict(self, e, y_prev):
        prev_output = [self.prev_intermediate_output[0], self.prev_intermediate_output[3]]
        input_vec = np.array([np.concatenate(([e, y_prev], prev_output))])
#         input_vec = np.array([[e, y_prev]])
#         print(input_vec)
        output = self.model.predict(input_vec, batch_size=1)
        
        self.prev_intermediate_output = self.intermediate_layer_model.predict(input_vec, batch_size=1).flatten()

        return output
        
    def copy_weights(self, source_model):
        for i in range(len(self.model.layers)):
            self.model.layers[i].set_weights(source_model.layers[i].get_weights())
    
    def temp_weight_change(self, layer_num, weight_num, index, delta):
        temp_weights = self.model.layers[layer_num].get_weights()
        self.stored_weights = self.model.layers[layer_num].get_weights()
        
        temp_weights[weight_num][index] += delta
        self.model.layers[layer_num].set_weights(temp_weights)

        self.stored_weight_layer = layer_num
    
    def temp_weight_restore(self):
        self.model.layers[self.stored_weight_layer].set_weights(self.stored_weights)

    
    def adjust_weights(self, weight_adjustment_list):
        current_w = 0        
        layers = self.model.layers
        for layer_num, layer in enumerate(layers):
            temp_weights = []
            for weight in layer.get_weights():
                for index in np.ndindex(weight.shape):
                    weight[index] += weight_adjustment_list[current_w]
                    current_w += 1
                temp_weights.append(weight)
                
            self.model.layers[layer_num].set_weights(temp_weights)
        
    def update_intermediate_model(self):
        self.prev_intermediate_output = np.array([0, 0, 0, 0])
        self.intermediate_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer("intermediate_layer").output)


# In[5]:


class PIDTuner:
    def __init__(self, servo_type, Kp, Ki, Kd):
        self.servo = Servo(servo_type, Kp, Ki, Kd)
        self.nn = NeuralNetwork()
        self.stored_nn = None
    
    def simulate(self, T, y0, r, d, angle_sample_rate, use_nn=True):
        self.servo.PID.reset()
        self.nn.update_intermediate_model()
        self.servo.PID.update_gains(0.32, 0.06, 0.00879)
        dy = np.zeros(len(T))
        y = np.zeros(len(T))
        angle = np.zeros(len(T))
        
        dy[0] = y0[0]
        y[0] = y0[1]
        
        angle[0] = y[0] * 180/math.pi + 180 + d[0]
#         print(angle)   
#         assert True == False
        Kp_list = np.zeros(len(T))
        Ki_list = np.zeros(len(T))
        Kd_list = np.zeros(len(T))
        
        sampled_angle = angle[0]

        for i in range(1, len(T)):

            if i % angle_sample_rate == 0:
                sampled_angle = angle[i-1]
                
            e = r[i] - sampled_angle
            
            if use_nn:
#                 e_norm = abs(e/180)
#                 y_prev_norm = abs((y[i-1]-(-math.pi))/(math.pi-(-math.pi)))
                e_norm = e/180
#                 y_prev_norm = (y[i-1]-(-math.pi))/(math.pi-(-math.pi))
                angle_norm = (sampled_angle - 180)/180

                Kp, Ki, Kd = np.absolute(self.nn.predict(e_norm, angle_norm).flatten())
#                 Kp, Ki, Kd = self.nn.predict(e_norm, angle_norm).flatten()

                Kp_list[i] = Kp
                Ki_list[i] = Ki
                Kd_list[i] = Kd
                assert not math.isnan(Kp) and not math.isnan(Ki) and not math.isnan(Kd), "NaN Neural Network output"
 
                self.servo.PID.update_gains(Kp, Ki, Kd)
            
            state, angle[i] = self.servo.simulate_step([dy[i-1], y[i-1]], sampled_angle, [T[i-1], T[i]], r[i], d[i])
            dy[i] = state[0]
            y[i] = state[1]
            
            
        return angle, (Kp_list, Ki_list, Kd_list)
    

    def compute_jacobian(self, T, y0, r, d, angle_sample_rate):
        jacobian = np.empty((len(T), self.nn.model.count_params()))
        current_w = 0
        y1, _ = self.simulate(T, y0, r, d, angle_sample_rate)

        
        layers = self.nn.model.layers
        for layer_num, layer in enumerate(layers):
            for weight_num, weight in enumerate(layer.get_weights()):
                for index in np.ndindex(weight.shape):

                    wij = weight[index]
                    epsilon = max(1, abs(wij)) * math.sqrt(sys.float_info.epsilon)        
                    
                    self.nn.temp_weight_change(layer_num, weight_num, index, epsilon)
        
                    y2, _ = self.simulate(T, y0, r, d, angle_sample_rate)
                    
                    self.nn.temp_weight_restore()
                    
                    for t in range(len(T)):
                        jacobian[t, current_w] = (y1[t] - y2[t])/epsilon
                    current_w += 1
            
        return jacobian, y1
    
    def store_nn(self):
        self.stored_nn = NeuralNetwork()
        self.stored_nn.copy_weights(self.nn.model)
        
    def restore_nn(self):
        self.nn = self.stored_nn


# In[6]:


def print_weights(neural_network):
    #To copy paste into C#
    weight_names = ['w1', 'b1', 'w2', 'b2']
    w_index = 0
    for layer in neural_network.model.layers:
        for w in layer.get_weights():
            print("Matrix<double>", weight_names[w_index], "= M.DenseOfArray(new double[,]{", end='')
            for i, row in enumerate(w):
                if type(row) is not np.float32:
                    print("{", end='')
                    for j in row:
                        print(str(j) + ",", end='')
                    print("},", end='')
                else:
                    if i == 0:
                        print("{",end='')
                    print(str(row) + ",", end='')
                    if i == len(w) - 1:
                        print("}",end='')
            print("});")
            w_index += 1


# In[9]:


aprbs_amp = 5
dt = 0.01
T, r, d, d_val, y0 = simulation_init(dt, 5.0, aprbs_hold=1.0, aprbs_amp=aprbs_amp, butter_cutoff=5)
angle_sample_rate = 0.08/dt #Sample every 80ms

max_trials = 100

sample_increase = 50
samples = sample_increase


best_mse = None
prev_sample_mse = None #For doings sample increase
trial = 0
rot_tuner = PIDTuner("rotation", 0.32, 0.06, 0.00879)

while(trial < max_trials):
    try:
        rot_tuner.nn = NeuralNetwork()
        prev_adj_mse = 0
        iters = 0
        damping_factor = 10
        print("Samples:", samples)
        print("Starting trial", trial)
        trial_finished = False
        convergence_double_check = False
        while(not trial_finished):

            #Compute jacobian
            J, y = rot_tuner.compute_jacobian(T[:samples], y0, r[:samples], d[:samples], angle_sample_rate) 

            #Get error of output without weight change
            error = y - r[:samples] 

            
            #Simulate whole signal before weight change
            y_full_sig, _ = rot_tuner.simulate(T, y0, r, d, angle_sample_rate) 
            y_full_sig_val, _ = rot_tuner.simulate(T, y0, r, d_val, angle_sample_rate)
            #Calculate mse of output without weight change
            mse_train = np.sum(np.square(y_full_sig - r))/len(T)
            mse = (mse_train + np.sum(np.square(y_full_sig_val - r))/len(T))/2
#             assert mse_train < aprbs_amp
            
            try_solve_levenberg = True
            while(try_solve_levenberg):
                #Compute adjustment amount
                A = np.matmul(np.transpose(J), J) + damping_factor * np.identity(J.shape[1])
                B = np.matmul(np.transpose(J), (error))
                try:
                    adjustment = np.matmul(np.linalg.inv(A), B) 
                    
                    #Store a copy of the current neural network before adjusting the weights
                    rot_tuner.store_nn()
                    rot_tuner.nn.adjust_weights(adjustment)

                    #Simulate with the adjusted weights
                    y_adj, _ = rot_tuner.simulate(T, y0, r, d, angle_sample_rate) 
                    y_adj_val, _ = rot_tuner.simulate(T, y0, r, d_val, angle_sample_rate)

                    #Calculate mse of output with weight change
                    adj_mse = (np.sum(np.square(y_adj - r))/len(T) + np.sum(np.square(y_adj_val - r))/len(T))/2
                    print("Iteration:", iters, "| mse:", mse, "| adj_mse:", adj_mse)

                    if adj_mse < mse:
                        #If squared error with weight change is better, half damping factor and keep new weights
                        damping_factor = damping_factor/2 
                        try_solve_levenberg = False #New sq error is less than the old one
                        
                        #Keep adjusting until our new adjustment performs worse
                        try_adjust = True
                        prev_adj_mse = adj_mse
                        while(try_adjust):
                            #Store a copy of the current neural network before adjusting the weights
                            rot_tuner.store_nn()
                            rot_tuner.nn.adjust_weights(adjustment)

                            #Simulate with the adjusted weights
                            y_adj, _ = rot_tuner.simulate(T, y0, r, d, angle_sample_rate) 
                            y_adj_val, _ = rot_tuner.simulate(T, y0, r, d_val, angle_sample_rate)

                            #Calculate mse of output with weight change
                            adj_mse = (np.sum(np.square(y_adj - r))/len(T) + np.sum(np.square(y_adj_val - r))/len(T))/2
                            if adj_mse >= prev_adj_mse:
                                print("Finished adjusting | prev_adj_mse:", prev_adj_mse, "| adj_mse:", adj_mse)
                                try_adjust = False
                                rot_tuner.restore_nn()
                            else:
                                print("Adjusting | adj_mse:", adj_mse)
                                prev_adj_mse = adj_mse
                    else:    
                        #If not better, double damping factor and restore old weights (from our stored copy)
                        damping_factor = 4 * damping_factor
                        rot_tuner.restore_nn()
                        
                    if (abs(1-adj_mse/prev_adj_mse) < 0.0001):
                        if convergence_double_check:
                            print("Trial finished, diff:", abs(1-adj_mse/prev_adj_mse))
                            trial_finished = True
                            try_solve_levenberg = False
                        else:
                            convergence_double_check = True
                    else:
                        convergence_double_check = False

                    prev_adj_mse = adj_mse
                    iters += 1
                except np.linalg.LinAlgError as err:
                    if 'Singular matrix' in str(err):
                        damping_factor = 4 * damping_factor
                        print("Unable to invert A")
                    else:
                        raise

        print_weights(rot_tuner.nn)

        if (best_mse is None or mse < best_mse) and mse != 0:
            print("Old best:", best_mse, "| New best:", mse)
            best_mse = mse
            best_model = NeuralNetwork()
            best_model.copy_weights(rot_tuner.nn.model)
            
            save_dict = {
                'model': best_model.model,
                'mse': best_mse
            }
            pkl.dump(save_dict, open('best_model_trial_' + str(trial) + '.pkl', 'wb'))
            
            
        if prev_sample_mse is None or mse < prev_sample_mse:
            if samples + sample_increase < len(T):
                samples += sample_increase
                print("Increasing samples")
            else:
                samples = len(T)
                
        prev_sample_mse = mse
        trial += 1
        
    except AssertionError:
        #If neural network outputs infinity or the mse is greater than the max amplitude of the APRBS
        print("Bad init, mse:", mse_train)


# In[10]:

print("Best model")
print_weights(best_model)
save_dict = {
  'model': best_model.model,
  'mse': best_mse
}
pkl.dump(save_dict, open('best_model_overall.pkl', 'wb'))




