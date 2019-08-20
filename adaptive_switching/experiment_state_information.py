"""
Explanation
"""
class state (object):
    """ interface to hold structure for subscriptions """
    def __init__(self):
        pass
    
    def update (self):
        pass
    
"""
================================================================================

                                Servo Classes

================================================================================
"""
class servo_info (state):
    """ abstract class to hold basic implementation for servo subscriptions"""
    def __init__(self):
        self.position = 0
        self.load = 0
        self.velocity = 0
        self.is_moving = 0
        
        self.normalized_position = 0
        self.normalized_load = 0
        self.normalized_velocity = 0
        
    def update(self, position, load, velocity, is_moving):
        self.position = position
        self.load = load
        self.velocity = velocity
        self.is_moving = is_moving
        
class gripper(servo_info):
    """ specific instance of gripper
        
        {Gripper}
        * Position : 0 - 1023
        * Velocity
        * Load : -1 : 1
        * Is Moving : 0 - 1
    
    """
    def update(self, position, load, velocity, is_moving):
        servo_info.update(self, position, load, velocity, is_moving)
        self.normalized_position = float(self.position)/1023.0
        self.normalized_load = float(self.load+0.56)*1.35
        self.normalized_velocity = None

class not_gripper(servo_info):
    """ specific instance of all servos other than gripper
        
        {Wrist Flexion}
        * Position : 0 - 4095
        * Velocity
        * Load : -1 : 1
        * Is Moving : 0 - 1
    """
    def update(self, position, load, velocity, is_moving):
        servo_info.update(self, position, load, velocity, is_moving)
#         self.normalized_position = float(self.position)/4095.0
        self.normalized_position = (float(self.position)-1.5)/3.22
        self.normalized_load = float(self.load+1)/2.0
        self.normalized_velocity = None

"""
================================================================================

                            Joint Activity Classes

================================================================================
"""


class joint_activity(state):
    
    def __init__(self):
        self.group = None
        self.joint_idx = None
        self.joint_id = None
        
    def update(self, joint_group, joint_idx, joint_id):
        self.joint_id = joint_id
        self.joint_idx = joint_idx
        self.group = joint_group

"""
===============================================================================

                                EMG Classes

===============================================================================
"""

class emg_activity(state):

    def __init__(self):
        self.emg1 = None
        self.emg2 = None
        self.emg3 = None

    def update(self, emg1, emg2, emg3):
        self.emg1 = emg1
        self.emg2 = emg2
        self.emg3 = emg3

class adc_activity(state):

    def __init__(self):
        self.signal0 = None
        self.signal1 = None
        self.signal2 = None

    def update(self, signal0, signal1, signal2):
        self.signal0 = signal0
        self.signal1 = signal1
        self.signal2 = signal2

# class parameter_activity(state):
#     def __init__(self):
#         self.gain1 = None
#         self.gain2 = None
#         self.gain3 = None
#
#     def update(self, gain1, gain2, gain3):
#         self.gain1 = gain1
#         self.gain2 = gain2
#         self.gain3 = gain3