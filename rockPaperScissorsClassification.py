# import all necessary libraries
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from os import listdir
import pandas as pd
from sklearn.model_selection import train_test_split

# declare constants
FOLDER_NAME = "asl_data\\asl_alphabet_train\\asl_alphabet_train\AA"
LABEL_LIST = ['A', 'B']
MAX_COUNTER = 7
SHOW_COORDS = False
SHOW_TRAINING = False
EPOCHS = 600

# declare list of data points to be transformed into pd df
data_list = []

# variables for mediapipe hand recognition
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# go through the folders to get every image and read it and show it with cv
folders = listdir(FOLDER_NAME)
for folder in folders:
    img_names = listdir(FOLDER_NAME + "\\" + folder)

    # goes through every image in the current folder
    for img_name in img_names:
        # declare landmark list to append to bigger list
        landmark_list = []

        # reads image using cv and transfers to correct colours
        img = cv2.imread(FOLDER_NAME + "\\" + folder + "\\" + img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # keep running the image through processing to get good coordinates
        counter = 0
        while counter < MAX_COUNTER:
            results = hands.process(img)
            counter += 1

        world_landmarks = results.multi_hand_world_landmarks
        multi_hand_landmarks = results.multi_hand_landmarks
        handedness = results.multi_handedness
        # draws the landmarks on the image
        if multi_hand_landmarks:
            for hand_landmark in multi_hand_landmarks:
                mpDraw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

        # prints coordinates if needed
        if SHOW_COORDS:
            if world_landmarks:
                print(world_landmarks)

        # shows the image on the screen and waits for user input if needed
        if SHOW_TRAINING:
            cv2.imshow("Image", img)
            cv2.waitKey(0)

        # put the handedness in the list
        if handedness:
            landmark_list.append(handedness[0].classification[0].index)

        # go through landmark obj and append the data to landmark list in order
        if world_landmarks:
            for landmarks in world_landmarks:
                landmarks = landmarks.landmark
                for landmark in landmarks:
                    landmark_list.append(landmark.x)
                    landmark_list.append(landmark.y)
                    landmark_list.append(landmark.z)

            # append the label to the end of the list
            landmark_list.append(folder)

            # append landmark list to the list of data
            data_list.append(landmark_list)

# convert data list into a pandas dataframe
# columns = ['handedness', 'wrist_x', 'wrist_y', 'thumb_cmc_x', 'thumb_cmc_y', 'thumb_mcp_x', 'thumb_mcp_y', 'thumb_ip_x',
#            'thumb_ip_y', 'thumb_tip_x', 'thumb_tip_y', 'index_mcp_x', 'index_mcp_y', 'index_pip_x', 'index_pip_y',
#            'index_dip_x', 'index_dip_y', 'index_tip_x', 'index_tip_y', 'middle_mcp_x', 'middle_mcp_y', 'middle_pip_x', 'middle_pip_y',
#            'middle_dip_x', 'middle_dip_y', 'middle_tip_x', 'middle_tip_y', 'ring_mcp_x', 'ring_mcp_y', 'ring_pip_x',
#            'ring_pip_y', 'ring_dip_x', 'ring_dip_y', 'ring_tip_x', 'ring_tip_y', 'pinky_mcp_x', 'pinky_mcp_y',
#            'pinky_pip_x', 'pinky_pip_y', 'pinky_dip_x', 'pinky_dip_y', 'pinky_tip_x', 'pinky_tip_y', 'label']
# numeric_cols = ['handedness', 'wrist_x', 'wrist_y', 'thumb_cmc_x', 'thumb_cmc_y', 'thumb_mcp_x', 'thumb_mcp_y', 'thumb_ip_x',
#                 'thumb_ip_y', 'thumb_tip_x', 'thumb_tip_y', 'index_mcp_x', 'index_mcp_y', 'index_pip_x', 'index_pip_y',
#                 'index_dip_x', 'index_dip_y', 'index_tip_x', 'index_tip_y', 'middle_mcp_x', 'middle_mcp_y',
#                 'middle_pip_x', 'middle_pip_y', 'middle_dip_x', 'middle_dip_y', 'middle_tip_x', 'middle_tip_y',
#                 'ring_mcp_x', 'ring_mcp_y', 'ring_pip_x', 'ring_pip_y', 'ring_dip_x', 'ring_dip_y', 'ring_tip_x',
#                 'ring_tip_y', 'pinky_mcp_x', 'pinky_mcp_y', 'pinky_pip_x', 'pinky_pip_y', 'pinky_dip_x', 'pinky_dip_y',
#                 'pinky_tip_x', 'pinky_tip_y']
columns = ['handedness', 'wrist_x', 'wrist_y', 'wrist_z', 'thumb_cmc_x', 'thumb_cmc_y', 'thumb_cmc_z', 'thumb_mcp_x', 'thumb_mcp_y', 'thumb_mcp_z', 'thumb_ip_x',
           'thumb_ip_y', 'thumb_ip_z', 'thumb_tip_x', 'thumb_tip_y', 'thumb_tip_z', 'index_mcp_x', 'index_mcp_y', 'index_mcp_z', 'index_pip_x', 'index_pip_y', 'index_pip_z',
           'index_dip_x', 'index_dip_y', 'index_dip_z', 'index_tip_x', 'index_tip_y', 'index_tip_z', 'middle_mcp_x', 'middle_mcp_y', 'middle_mcp_z', 'middle_pip_x', 'middle_pip_y', 'middle_pip_z',
           'middle_dip_x', 'middle_dip_y', 'middle_dip_z', 'middle_tip_x', 'middle_tip_y', 'middle_tip_z', 'ring_mcp_x', 'ring_mcp_y', 'ring_mcp_z', 'ring_pip_x',
           'ring_pip_y', 'ring_pip_z', 'ring_dip_x', 'ring_dip_y', 'ring_dip_z', 'ring_tip_x', 'ring_tip_y', 'ring_tip_z', 'pinky_mcp_x', 'pinky_mcp_y', 'pinky_mcp_z',
           'pinky_pip_x', 'pinky_pip_y', 'pinky_pip_z', 'pinky_dip_x', 'pinky_dip_y', 'pinky_dip_z', 'pinky_tip_x', 'pinky_tip_y', 'pinky_tip_z', 'label']
numeric_cols = ['handedness', 'wrist_x', 'wrist_y', 'wrist_z', 'thumb_cmc_x', 'thumb_cmc_y', 'thumb_cmc_z', 'thumb_mcp_x', 'thumb_mcp_y', 'thumb_mcp_z', 'thumb_ip_x',
                'thumb_ip_y', 'thumb_ip_z', 'thumb_tip_x', 'thumb_tip_y', 'thumb_tip_z', 'index_mcp_x', 'index_mcp_y', 'index_mcp_z', 'index_pip_x', 'index_pip_y', 'index_pip_z',
                'index_dip_x', 'index_dip_y', 'index_dip_z', 'index_tip_x', 'index_tip_y', 'index_tip_z', 'middle_mcp_x', 'middle_mcp_y', 'middle_mcp_z', 'middle_pip_x', 'middle_pip_y', 'middle_pip_z',
                'middle_dip_x', 'middle_dip_y', 'middle_dip_z', 'middle_tip_x', 'middle_tip_y', 'middle_tip_z', 'ring_mcp_x', 'ring_mcp_y', 'ring_mcp_z', 'ring_pip_x',
                'ring_pip_y', 'ring_pip_z', 'ring_dip_x', 'ring_dip_y', 'ring_dip_z', 'ring_tip_x', 'ring_tip_y', 'ring_tip_z', 'pinky_mcp_x', 'pinky_mcp_y', 'pinky_mcp_z',
                'pinky_pip_x', 'pinky_pip_y', 'pinky_pip_z', 'pinky_dip_x', 'pinky_dip_y', 'pinky_dip_z', 'pinky_tip_x', 'pinky_tip_y', 'pinky_tip_z']

df = pd.DataFrame(data_list, columns=columns)
df = df.drop(df[df['handedness'] == 'none'].index)
df = df.drop(df[df['handedness'] == 'paper'].index)

df['label'].replace(['A', 'B', 'C'], [0, 1, 2], inplace=True)
print(df.dtypes)
print(df)
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col])

df['label'] = pd.to_numeric(df['label'])
print(df.dtypes)


# create the test and training data
y = df.pop('label')
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)

# create the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(((21 * 3) + 1, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(60, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# train model
model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=128,
    validation_data=(x_test, y_test)
)

# evaluate model
scores_train = model.evaluate(x_train, y_train, verbose=0)
print("Accuracy for training data " + str(scores_train[1]))

scores_test = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy for test data " + str(scores_test[1]))

# save the model
model.save(FOLDER_NAME + "_model")
