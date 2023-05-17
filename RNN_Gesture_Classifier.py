from cProfile import label
from calendar import c
from cmath import pi
from gettext import install
from math import atan2
from operator import delitem
import cv2
from cv2 import circle 
import mediapipe as mp 
import time
from sklearn import pipeline
from sklearn.metrics import classification_report
from sympy import im 
from send_OSC import SendOSC, Landmark
from pythonosc import udp_client
import matplotlib.pyplot as plt
import pandas as pd

import argparse

from pythonosc import dispatcher
from pythonosc import osc_server
import csv
import numpy as np
from regex import B

import numpy as np
import csv
import os
mp_holistic = mp.solutions.holistic 
mp_pose = mp.solutions.pose 
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class_labels = ['Up','Down']

def create_vector_csv():
    # Create headings for CSV file
    headings = ['Class']
    # 33 landmarks for pose
    for val in range(1,34):
        headings += ['X{}'.format(val),'Y{}'.format(val),'Z{}'.format(val),'V{}'.format(val)]
    # 21 for rh and lh
    for val in range(1,22):
        headings += ['XL{}'.format(val),'YL{}'.format(val),'ZL{}'.format(val)]
    for val in range(1,22):
        headings += ['XR{}'.format(val),'YR{}'.format(val),'ZR{}'.format(val)]

    # Write csv file
    with open('landmark_vector_data.csv', mode = 'w', newline = '') as file:
        write_csv = csv.writer(file, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        write_csv.writerow(headings)
    return 
#create_vector_csv()

def export_vector_to_csv(input,user,gesture_class, gesture_length, sequence_length, no_sequences, wait_time):
    """
    input: video input to holistic classification
    user: name of user
    gesture_class: name of gesture class captured
    gesture_length: the number of vectors in one gesture
    wait_time: the time in seconds between each sequence collection
    sequence_length: the sequence length that includes the gesture length and additional padding
    no_sequences: the number of sequences to be captured for the specific class
    allows for padding 0s to ensure all classes have same sequence length 
    """

    # Get webcam input
    width = 1280 #800, 1024, 1920 
    height = 720 #600, 780, 1080

    cap = cv2.VideoCapture(input) # 'Videos/no-leg-ball.mp4'
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 

    path = 'Class-Images'

    frame = 0
    
    # Begin new instance of mediapipe feed
    with mp_holistic.Holistic(
        model_complexity = 1,
        min_detection_confidence = 0.8,
        min_tracking_confidence = 0.8) as holistic:
        toc = time.perf_counter()
        i=0
              
        # loop through number of sequences for each class
        for sequence in range(no_sequences):

            # loop through sequence length (number of frames per sequence)            
            for frame_num in range(gesture_length):

                isTrue, image = cap.read() 
                if not isTrue:
                    print("Empty camera frame")
                    continue 
                # Improve performance 
                image.flags.writeable = False
                # Recolour image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Make detection and store in output array
                output = holistic.process(image)
                image.flags.writeable = True
                # Recolour to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                h,w,c = image.shape
                
                # Draw landmarks on image
                mp_draw.draw_landmarks(
                    image,
                    output.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks
                
                mp_draw.draw_landmarks(
                    image,
                    output.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks

                mp_draw.draw_landmarks(
                    image,
                    output.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks

                # Extract available landmarks
                # for lm in mp_pose.PoseLandmark: print(lm)
                try:
                    # extract normalised coordinates
                    pose_norm = np.array([[res.x, res.y, res.z, res.visibility] for res in output.pose_landmarks.landmark]).flatten() if output.pose_landmarks else np.zeros(132)
                    lh_norm = np.array([[res.x, res.y, res.z] for res in output.left_hand_landmarks.landmark]).flatten() if output.left_hand_landmarks else np.zeros(21*3)
                    rh_norm = np.array([[res.x, res.y, res.z] for res in output.right_hand_landmarks.landmark]).flatten() if output.right_hand_landmarks else np.zeros(21*3)
                    
                    # extract world coordinates
                    pose_world = np.array([[res.x, res.y, res.z, res.visibility] for res in output.pose_world_landmarks.landmark]).flatten() if output.pose_world_landmarks else np.zeros(132)

                    # create vector of landmark coordinates
                    holistic_row = list(np.concatenate([pose_world,lh_norm,rh_norm]))
                    # Append class name
                    holistic_row.insert(0,gesture_class)
                    
                    image = cv2.flip(image,1)
                except:
                        pass # pass if landmarks detected

                # wait for t seconds after of each sequence capture
                if frame_num == 0:   
                    # countdown from wait_time seconds to 1
                    for t in range(wait_time, 0, -1):
                        # display the countdown text on the frozen image
                        countdown_img = image.copy()
                        cv2.putText(countdown_img, 'COLLECTING IN ' + str(t), (120, 200), 
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(countdown_img, 'Class: {}, Video: {}'.format(gesture_class, sequence), (30,50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow("Countdown", countdown_img)
                        cv2.waitKey(1000)  # wait for 1 second

                    # close the countdown window
                    cv2.destroyWindow("Countdown")

                else: 
                    cv2.putText(image, 'COLLECT', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Class: {}, Video: {}'.format(gesture_class, sequence), (30,50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Vector: {} / {}'.format(frame_num, gesture_length), (400,300), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # write vector to csv file
                cv2.imwrite(os.path.join(path,'{}'.format(gesture_class) + '-{}'.format(user) + '-{}'.format(i)  + '.png'),image)
                with open('landmark_vector_data.csv', mode = 'a', newline = '') as file:
                    write_csv = csv.writer(file, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                    write_csv.writerow(holistic_row)

                frame+=1
                print('frame: ',frame)

                print('sequence number: ',sequence)        

                # break loop and close windows if q key is pressed
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            # preprocess data by padding with zeros after each sequence
            padding = np.zeros(shape=((sequence_length-gesture_length),len(holistic_row)),dtype=object)
            # first column of array filled by class name
            padding[:,0]=[str(gesture_class)]
            with open('landmark_vector_data.csv', mode = 'a', newline = '') as file:
                write_csv = csv.writer(file, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
                write_csv.writerows(padding)             
    return 
#export_vector_to_csv(1,'Ben','Up',20,40,80,2)

sequence_length = 40
window_size = 30
stride = 5

def train_LSTM(sequence_length,window_size,stride):
    """
    train_LSTM is a function trains an LSTM network and saves its result as a .pkl file
    ...
    sequence_length: the sequence length that includes the gesture length and additional padding
    window_size: sliding window technique provides a way to transform variable-length sequences\\
        into fixed-length input sequences by sliding a window over the data. This ensures that \\
            the LSTM model can handle inputs of consistent lengths.
    stride: number of steps by which the sliding window moves forward after each iteration, \\
        determining the overlap or gap between consecutive input sequences.
    """
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Flatten, Reshape, TimeDistributed, Dropout
    from keras.utils import to_categorical
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    import pickle 

    # 1. Prepare the dataset
    # Your dataset of N sequences with 100 time steps of landmark datapoints
    label_encoder = LabelEncoder()

    df = pd.read_csv('landmark_vector_data.csv') 
    df = df.fillna(0)
    # Remove class oclumn 
    data = df.drop('Class',axis=1) #features
    labels = df['Class'] # target
    # Encode the class labels 
    labels = label_encoder.fit_transform(np.array(labels))
    labels = to_categorical(labels)
    # Reshape into 3D array
    data = np.array(data).reshape(-1,sequence_length, data.shape[1])
    labels = np.array(labels).reshape(-1,sequence_length,labels.shape[1])

    # 2. Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # 3. Perform sliding window technique 
    def sliding_window(data, labels, window_size):
        X = []
        y = []
        for i in range(len(data)):
            for j in range(0, len(data[i])-window_size, stride):
                X.append(data[i][j:j+window_size])
                y.append(labels[i][j:j+window_size])
        return np.array(X), np.array(y)

    # Create new dataset based on sliding window
    x_train_sliding, y_train_sliding = sliding_window(x_train, y_train, window_size)
    x_test_sliding, y_test_sliding = sliding_window(x_test, y_test, window_size)

    # 4. Define the RNN model
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(window_size, data.shape[2]), return_sequences=True))
    model.add(TimeDistributed(Dense(units=labels.shape[2], activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train_sliding, y_train_sliding, validation_data=(x_test_sliding, y_test_sliding),epochs=10, batch_size=16) 
    
    """ 
    #without sliding window technique
    sequence_length=40

    # 4. Define the RNN model
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(sequence_length, data.shape[2]), return_sequences=True))
    model.add(Dropout(0.5))  # Add dropout regularization to the LSTM layer
    model.add(TimeDistributed(Dense(units=labels.shape[2], activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=10, batch_size=16) 
    """

    # 6. Make predictions
    y_predictions = model.predict(x_test)
    #print(y_predictions)

    # 7. Evaluate model
    a=np.argmax(y_test,axis=2).reshape(-1,1)
    b=np.argmax(y_predictions,axis=2).reshape(-1,1)
    report = classification_report(a,b, output_dict=True)    
    print(report)
    
    # 8. Write binary pkl file and export lstm model
    with open('lstm_model.pkl','wb') as file:
        pickle.dump(model,file) 
        
    return
train_LSTM(sequence_length,window_size,stride)

def send_RNN_landmark_class():
    import pickle
    
    # read model 
    with open('lstm_model.pkl','rb') as file:
        model = pickle.load(file)

    # initialise n x m array used as input to LSTM model    
    feature_size = 258
    input_array = np.zeros((window_size,feature_size)) 
    # Reshape into 3D array 
    input_array = input_array.reshape(-1,input_array.shape[0],input_array.shape[1])

    # Get webcam input
    cap = cv2.VideoCapture(1) # 'VideosSelf/Arms-Raise-Front.mov' 'Videos/no-leg-ball.mp4'
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #800, 1024, 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #600, 780, 1080

    # Begin new instance of mediapipe feed
    with mp_holistic.Holistic(
        model_complexity = 2,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as pose:

        frame = 0

        while cap.isOpened():

            isTrue, image = cap.read() 

            if not isTrue:
                print("Empty camera frame")
                continue 

            # Improve performance 
            image.flags.writeable = False
            # Recolour image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Make detection and store in output array
            output = pose.process(image)
            image.flags.writeable = True
            # Recolour to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks on image
            mp_draw.draw_landmarks(
                image,
                output.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks
            
            mp_draw.draw_landmarks(
                image,
                output.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks

            mp_draw.draw_landmarks(
                image,
                output.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks

            # Extract available landmarks
            try:
                # extract normalised coordinates 
                pose_norm = output.pose_landmarks.landmark
                lh_norm = output.left_hand_landmarks.landmark
                rh_norm = output.right_hand_landmarks.landmark

                pose_world = output.pose_world_landmarks.landmark

                # normalised row
                pose_row = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_norm]).flatten() if output.pose_landmarks else np.zeros(132)
                lh_row = np.array([[res.x, res.y, res.z] for res in lh_norm]).flatten() if output.left_hand_landmarks else np.zeros(21*3)
                rh_row = np.array([[res.x, res.y, res.z] for res in rh_norm]).flatten() if output.right_hand_landmarks else np.zeros(21*3)
                
                # extract world coordinates
                pose_world_row = np.array([[res.x, res.y, res.z, res.visibility] for res in pose_world]).flatten() if output.pose_world_landmarks else np.zeros(132)

                # create list of landmark coordinates
                holistic_row = list(np.concatenate([pose_world_row,lh_row,rh_row]))
                pose_vector = pd.DataFrame([holistic_row])
                
                # Begin processing data array to input to RNN model
                
                # Deleting first row of gesture array
                input_array = np.delete(input_array[0],0, axis=0)

                # Add incoming vector to new row to create n x m input array
                input_array = np.vstack((input_array,pose_vector))

                # Reshape into 3D array 
                input_array = input_array.reshape(-1,input_array.shape[0],input_array.shape[1])

                # Process the input array
                output_prob = model.predict(input_array)
                print(output_prob)

                # Get index of highest gesture probability
                ind_max = np.unravel_index(np.argmax(output_prob, axis=None), output_prob.shape)
                gesture_predict = class_labels[ind_max[2]]
                # Extract highest gesture probability 
                gesture_prob = output_prob[ind_max]
                print(gesture_predict, ': ', gesture_prob)

            except:
                pass # pass if landmarks detected
            
            frame += 1             

            # Display 
            # Mirror image for webcam display
            image = cv2.flip(image,1)

            try:
                # Display class prediction
                cv2.putText(image, gesture_predict, (540, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),2)
            except:
                pass
            
            # Visualise image
            cv2.imshow('Video', image)

            # break loop and close windows if q key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    cap.release()
    cv2.destroyAllWindows
#send_RNN_landmark_class()

