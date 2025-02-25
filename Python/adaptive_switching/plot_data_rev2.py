import pickle
import matplotlib                   # required for plotting
import matplotlib.pyplot as plt     # required for plotting

# Load the data from test_data file that was generated by adaptive_switching.py

try:
    loaded_data = open('test_data.txt', 'rb')
    list_step, list_shoulder_state, list_shoulder_pred, list_wristFlex_state, list_wristFlex_pred, list_hand_state, list_hand_pred = pickle.load(loaded_data)
    #list_step, list_shoulder_state, list_shoulder_pred = pickle.load(loaded_data)
    loaded_data.close()

    fig, ax = plt.subplots()


    ax.plot(list_step, list_shoulder_state, color = '#2471a3' , label='Shoulder State')
    ax.plot(list_step, list_shoulder_pred, color = '#a9cce3', label='Shoulder Pred')
    ax.plot(list_step, list_wristFlex_state, color = '#a93226' , label='wristFlex State')
    ax.plot(list_step, list_wristFlex_pred, color = '#e6b0aa', label='wristFlex Pred')
    ax.plot(list_step, list_hand_state, color = '#1e8449', label='Hand State')
    ax.plot(list_step, list_hand_pred, color = '#a9dfbf', label='Hand Pred')
    ax.legend(loc='upper left')
    ax.set(xlabel='Step Number', ylabel='Normalized Value',
    title='Predictions vs Step Number')
    ax.grid()
    plt.show()
    
except (OSError, IOError) as e:
    print('Pickle file does not exist yet')







