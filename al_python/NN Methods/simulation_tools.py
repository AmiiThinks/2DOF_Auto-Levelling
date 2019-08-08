import math
import numpy as np
from scipy.integrate import odeint
from scipy import signal


class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
#         self.I_max = 5
        self.e_prev = 0
        self.t_prev = 0
        self.P = 0
        self.I = 0
        self.D = 0
        self.u = 0

    def update(self, e, t):
        
        delta_time = t - self.t_prev
        delta_error =  e - self.e_prev
        
        self.P = e
    
        self.I += e * delta_time

#         if (self.I < -self.I_max):
#             self.I = -self.I_max
#         elif (self.I > self.I_max):
#             self.I = self.I_max

        if delta_time > 0:
            self.D = delta_error / delta_time
        else:
            self.D = 0

        self.t_prev = t
        self.e_prev = e
        self.u = self.Kp * self.P + self.Ki * self.I + self.Kd * self.D


        return self.u 
       
    def reset(self):
        self.e_prev = 0
        self.t_prev = 0
        self.P = 0
        self.I = 0
        self.D = 0
        self.u = 0

        
    def update_gains(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

class Servo:
    def __init__(self, servo_type, Kp, Ki, Kd):
        R = 8.3 #Motor resistance (ohms)
        L = 2.03e-3 #Motor inductance (H)
        gear_ratio = 193 #Motor gear ratio
        gear_eff = 0.836 #Motor gear efficiency
        K_w = 93.1 #Speed constant (rad/V)
        K_t = 0.0107 #Torque constant (Nm/A)
        J_m = 8.68e-8 #Motor moment of inertia (kgm^2)
        b_m = 8.87e-8 #Motor friction (Nms)

        P_m = 32 #Internal motor P value
        I_m = 0 #Internal motor I value
        D_m = 0 #Internal motor D value
        Vs = 12 #Source voltage V

        Kp_m = P_m/8 * 2048 * Vs/(511*math.pi) #Motor Kp
        Ki_m = 1000 * I_m / 2048 * 2048 * Vs/(511*math.pi) #Motor Ki
        Kd_m = 4 * D_m / 1000 * 2048 * Vs/(511*math.pi) #Motor Kd

        m = 0.077 #Servo mass (kg)
        b = 0.0355 #Servo width (m)
        h = 0.0244 #Servo height (m)
        d = 0.033 + 0.012 #Servo depth (m)
        bracket_L = 0.03 #Bracket lenght (m)
        gripper_com = 0.03 #Centre of mass of gripper wrt where bracket connects

        if servo_type == "rotation":
            J_l = (2 * m * (b**2 + h**2)/12)
        elif servo_type == "flexion":
            J_l = m * (d**2/3 + d * bracket_L + bracket_L**2)
        else:
            print("No type specified")
            J_l = 0
        #Simplified tf (eq 31)
        c1 = gear_ratio * gear_eff * K_t
        c2 = gear_ratio**2 * gear_eff
        n1 = c1 * Kd_m
        n2 = c1 * Kp_m
        n3 = c1 * Ki_m

        d1 = R * (J_l + J_m * c2)
        d2 = b_m * c2 * R + K_t * c2 / K_w + c1 * Kd_m
        d3 = c1 * Kp_m
        d4 = c1 * Ki_m
        den = [d1, d2, d3, d4]

        self.A = -d2/d1
        self.B = -d3/d1
        self.C = n2/d1
        
        self.PID = PID(Kp, Ki, Kd)
       
        
    #state = [dy, y]
    def ode(self, y, t, u):
        return (self.A*y[0] + self.B*y[1] + self.C*u, y[0]) #return [ddy, dy]
    
    def simulate_step(self, y0, cur_angle, t, r, d, use_sat=True):
        e = r - cur_angle #calculate error from setpoint (degrees)
        u = self.PID.update(e, t[1]) #get control signal by setting error and current time (t = [t_prev, t_cur])
        
        #tick saturation limit is 130
        #convert to deg 130/11.3611111111 = 11.4425
        if use_sat:
            u = min(max(u, -11.4425), 11.4425)

        u = u * math.pi/180 + y0[1] #control signal is added to previous angle, rather than just setting the angle (convert to radians)

        _, state = odeint(self.ode, y0, t, args=(u,)) #simulate step using odeint
        
        next_angle = state[1] * 180/math.pi + 180 + d #Convert back to degrees and add 180 degrees and disturbance to get phi/theta

        return state, next_angle #return the state of the servo (angular velocity and angular position) and the angle in degrees
        
class PRBS:
    def __init__(self):
        self.polynomials = {
            '2': [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            '3': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            '4': [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            '5': [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            '6': [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            '7': [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            '8': [0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
            '9': [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
            }

    def find_optimal_grade_and_scale(self, l, s_max):        
        l_max = 1
        n = 1
        n_opt = 1
        l_best = None
        while l_max <= l:

            l_max = 2**n - 1 #Paper is wrong it says n^2
            l_current = l_max * math.floor(s_max/n + 1/2)

            if l_best is None or (abs(l_current - l) < l_best):
                l_best = abs(l_current - l) #also wrong here
                n_opt = n
            n += 1

        k_opt = s_max/n_opt

        return n_opt, k_opt

    def generate_PRBS(self, l, s_max):
        y = np.empty(l)
        n_opt, k_opt = self.find_optimal_grade_and_scale(l, s_max)
        k_opt = round(k_opt)

        shift_register = [1] * n_opt
        polynomial = self.polynomials[str(n_opt)]
        intervals = 0


        for i in range(len(y)):
            #set new value if multiple of k_opt (same as scaling the signal by k_opt)
            if i % k_opt == 0:
                y[i] = shift_register[-1]
                if y[i] != y[i-1]:
                    intervals += 1


                x = 0
                vals_to_xor = [j-1 for j, p in enumerate(polynomial) if p == 1] #indexes start at 1

                for c in vals_to_xor:
                    x = int(x != shift_register[c])

                shift_register.pop()
                shift_register.insert(0, x)

            else:
                y[i] = y[i-1] #hold value

        return y, intervals

    #generates APRBS centered at 0
    #amp is peak to peak
    def generate_APRBS(self, l, s_max, amp):

        y, intervals = self.generate_PRBS(l, s_max)

        y_max = amp/2
        y_min = -y_max

        amplitude_step_size = (y_max - y_min)/(intervals-1)

        #generate random amplitudes
        new_amplitudes = np.empty(intervals)
        samples = np.arange(0, intervals)
        np.random.shuffle(samples)
        for i in range(intervals):
            new_amplitudes[i] = samples[i] * amplitude_step_size

        #apply amplitude modification to intervals in y
        y_amp_mod = np.empty(len(y))
        amp_index = 0
        for i in range(len(y)):
            if i != 0 and y[i] != y[i-1]:
                amp_index += 1
            y_amp_mod[i] = new_amplitudes[amp_index] - (y_max - y_min)/2 #re-centre the signal
        return y_amp_mod
   
    def apply_butter(self, y):
        fc = 10# Cut-off frequency of the filter
        fs = 1/0.005
        w = fc / (fs / 2) # Normalize the frequency
        b, a = signal.butter(1, w, 'low')
        return signal.filtfilt(b, a, y)
            
def simulation_init(time_step, length, aprbs_hold, aprbs_amp):
    T = np.arange(0, length, time_step)
    r = np.array(T)
    r[:] = 180
    
    sig_gen = PRBS()
    d = sig_gen.apply_butter(sig_gen.generate_APRBS(len(T), aprbs_hold, aprbs_amp))
    
    d_test = np.array(T)
    d_test[:] = 0

    up = len(T)//4
    down = len(T)//4 * 2
    d_test[up:down] = 45
   
    y0 = [0, 0]
    
    return T, r, d, d_test ,y0
