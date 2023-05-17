# IAAI

Gesture recognition algorithm that infers temporal data using an LSTM network.

The model is trained on a modest dataset of two gestures, each 200 sequences, with each sequence 40 vectors in length, and 258 landmark position datapoints per sequence, inferred from the mediaipe holistic model and saved to a .csv file

The model uses Mediapipe Hollistic (Hands and Pose) to infer 258 landmark coordinates - 33 pose landmarks each in x,y,z and visisbility and 21 landmark coordinates in x,y,z for each hand.

The first gesture consists of moving the left hand in an upwards from the waist to above the head. 
The second gesture consists of moving the left hand downwards from above the head to below the waist.

Each gesture is either 20 or 30 vectors in length, and padded with 0s to fill the sequence length to 40 vectors.

The gestures used are a crude example to demonstrate if the algorithm can accurately distinguish between the gestures in real-time.


