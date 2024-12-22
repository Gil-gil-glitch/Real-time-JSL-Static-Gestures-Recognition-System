import pickle
import math
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from pathlib import Path


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands 



#loading our kNN model

with open("knn_model.pkl", "rb") as model_file:

    knn_model = pickle.load(model_file)


# gesture maps based on the encoding of our kNN model
gesture_mapping = {
    0: "あ",
    1: "い",
    2: "う",
    3: "え",
    4: "お",
    5: "か",
    6: "き",
    7: "く",
    8: "け",
    9: "こ",
    10: "さ",
    11: "し",
    12: "す",
    13: "せ",
    14: "そ",
    15: "た",
    16: "ち",
    17: "つ",
    18: "て",
    19: "と",
    20: "な",
    21: "に",
    22: "ぬ",
    23: "ね",
    24: "は",
    25: "ひ",
    26: "ふ",
    27: "へ",
    28: "ほ",
    29: "ま",
    30: "み",
    31: "む",
    32: "め",
    33: "や",
    34: "ゆ",
    35: "よ",
    36: "ら",
    37: "る",
    38: "れ",
    39: "ろ",
    40: "わ",
    
   
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


###### 3D SCATTERPLOT SETUP ###############
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_zlim(-0.1, 1.1)

ax.view_init(elev=180, azim=90)



####### VARIABLES ###########

scatter = ax.scatter([], [], [], color='red', s=50)
lines = [ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in HAND_CONNECTIONS]


data = []
recording = False
calibrated = False
frame_count = 0
MAX_FRAMES = 30
label = None
label_entered = False
calibration_data = None  # storing the open-palm calibration coordinates

####### FUNCTIONS ##########

# calibrate by setting fixed min and max values for normalization
def calibrate_hand(landmarks):
    global min_x_calib, max_x_calib, min_y_calib, max_y_calib, min_z_calib, max_z_calib
    
    x_values = [lm.x for lm in landmarks]
    
    y_values = [lm.y for lm in landmarks]
    
    z_values = [lm.z for lm in landmarks]
    
    min_x_calib, max_x_calib = min(x_values), max(x_values)
    
    min_y_calib, max_y_calib = min(y_values), max(y_values)
    
    min_z_calib, max_z_calib = min(z_values), max(z_values)
    
    print("Calibration complete. Min and max values set.")

# normalize landmarks based on fixed calibration min and max values
def normalize_landmarks(landmarks):
    
    normalized_landmarks = []
    
    for lm in landmarks:
    
        norm_x = (lm.x - min_x_calib) / (max_x_calib - min_x_calib) if max_x_calib != min_x_calib else 0
        
        norm_y = (lm.y - min_y_calib) / (max_y_calib - min_y_calib) if max_y_calib != min_y_calib else 0
        
        norm_z = (lm.z - min_z_calib) / (max_z_calib - min_z_calib) if max_z_calib != min_z_calib else 0
        
        normalized_landmarks.append((norm_x, norm_y, norm_z))
    
    return normalized_landmarks

def calculate_distance(landmarks):
    """
    Calculates the distance between all pairs of landmarks, starting from landmark 0. 
    Note that it excludes redundant pairings like landmark 4 to 3 since the distance
    with landmark 3 to 4 already exists.
    
    Formula:

        Distance = √[(xᵢ - xⱼ)² + (yᵢ - yⱼ)² + (zᵢ - zⱼ)²]

    """
    frame_data = {}

    num_landmarks = len(landmarks)

    for i in range(num_landmarks):
        
        for j in range(i + 1, num_landmarks):
        
            x1, y1, z1 = landmarks[i]
        
            x2, y2, z2 = landmarks[j]

            # calculate the Euclidean distance
            x_difference = (x1 - x2) ** 2
        
            y_difference = (y1 - y2) ** 2
        
            z_difference = (z1 - z2) ** 2

            distance = math.sqrt(x_difference + y_difference + z_difference)

            frame_data[f"Landmark_{i}_to_Landmark_{j}_distance"] = distance # store the distance in the dictionary with a unique key (e.g., "Landmark_0_to_Landmark_1")

    return frame_data

# calculating angles
def calculate_angles_between_pairs(landmarks):
    """
    Calculates the angle between each pair of landmarks relative to the three main axes (x, y, z),
    starting from landmark 0. Similar to calculate_distance, the code does not account for 
    redundant pairings.

    The following formula is used: 

    Angle = arccos[(xᵢ * xⱼ + yᵢ * yⱼ + zᵢ * zⱼ) / (√(xᵢ² + yᵢ² + zᵢ²) * √(xⱼ² + yⱼ² + zⱼ²))]
    """
    num_landmarks = len(landmarks)
    
    angle_data = {}

    # iterate through each unique pair of landmarks, including starting from landmark 0
    for i in range(num_landmarks):
    
        for j in range(i + 1, num_landmarks):
    
            x1, y1, z1 = landmarks[i]
    
            x2, y2, z2 = landmarks[j]

            # calculate the vector components between the two landmarks
            dx = x2 - x1
    
            dy = y2 - y1
    
            dz = z2 - z1

            # calculate the magnitude of the vector
            magnitude = math.sqrt(dx**2 + dy**2 + dz**2)

            # avoid division by zero
            if magnitude == 0:
    
                continue

    
            # calculate angles in degrees for each axis
    
            angle_x = math.degrees(math.acos(dx / magnitude)) if magnitude != 0 else 0
    
            angle_y = math.degrees(math.acos(dy / magnitude)) if magnitude != 0 else 0
    
            angle_z = math.degrees(math.acos(dz / magnitude)) if magnitude != 0 else 0

            # store the angles in the dictionary with a unique key
    
            angle_data[f"Landmark_{i}_to_Landmark_{j}_angle_x"] = angle_x
    
            angle_data[f"Landmark_{i}_to_Landmark_{j}_angle_y"] = angle_y
    
            angle_data[f"Landmark_{i}_to_Landmark_{j}_angle_z"] = angle_z

    return angle_data


# Calculating direction
def calculate_direction(landmarks):
    """
    Calculates the direction vectors of the 5 main fingers relative to the wrist (landmark 0).
    The direction is represented in the x, y, and z axes.
    """
    
    # indices of the tips of the 5 main finger tips
    finger_tips = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20
    }

    directions = {}

    # gets the wrist landmark (reference point)
    wrist = landmarks[0]

    for finger, tip_index in finger_tips.items():
    
        tip = landmarks[tip_index]
        
    
        # calculate the direction vector components (dx, dy, dz)
    
        direction_x = tip[0] - wrist[0]
    
        direction_y = tip[1] - wrist[1]
    
        direction_z = tip[2] - wrist[2]

    
        directions[f"{finger}_direction_x"] = direction_x
    
        directions[f"{finger}_direction_y"] = direction_y
    
        directions[f"{finger}_direction_z"] = direction_z

    return directions

#unimportant_features is a list of the non-significant features (score of 0) identified by Random Forest.
unimportant_features = [
    "index_direction_z",
    "Landmark_0_to_Landmark_6_distance",
    "Landmark_7_to_Landmark_13_angle_x",
    "Landmark_0_to_Landmark_8_distance",
    "Landmark_2_to_Landmark_11_angle_z",
    "Landmark_5_to_Landmark_14_angle_z",
    "index_direction_x",
    "Landmark_10_to_Landmark_16_angle_z",
    "Landmark_1_to_Landmark_4_distance",
    "Landmark_5_to_Landmark_14_distance",
    "Landmark_0_to_Landmark_5_angle_x",
    "Landmark_0_to_Landmark_3_angle_z",
    "Landmark_14_to_Landmark_17_angle_x",
    "Landmark_15_to_Landmark_16_distance",
    "Landmark_8_to_Landmark_14_angle_x"


]

important_features = [
    "Landmark_7_to_Landmark_14_angle_y",
    "Landmark_12_to_Landmark_20_angle_y",
    "Landmark_3_to_Landmark_5_angle_x",
    "ring_direction_z",
    "Landmark_4_to_Landmark_6_angle_x",
    "Landmark_7_to_Landmark_19_distance",
    "Landmark_2_to_Landmark_11_distance",
    "Landmark_5_to_Landmark_8_distance",
    "Landmark_11_to_Landmark_17_angle_y",
    "Landmark_3_to_Landmark_8_angle_z",
    "Landmark_4_to_Landmark_6_angle_y",
    "Landmark_11_to_Landmark_18_angle_x",
    "Landmark_4_to_Landmark_15_angle_y",
    "Landmark_7_to_Landmark_13_distance",
    "Landmark_6_to_Landmark_7_angle_y",
    "Landmark_6_to_Landmark_20_distance",
    "Landmark_4_to_Landmark_10_angle_y",
    "Landmark_9_to_Landmark_12_angle_y",
    "Landmark_5_to_Landmark_6_angle_x",
    "Landmark_3_to_Landmark_10_angle_y",
    "Landmark_11_to_Landmark_15_angle_y",
    "Landmark_2_to_Landmark_18_angle_x",
    "Landmark_18_to_Landmark_20_distance",
    "Landmark_2_to_Landmark_11_angle_y",
    "Landmark_8_to_Landmark_15_angle_z",
    "Landmark_12_to_Landmark_14_angle_x",
    "Landmark_0_to_Landmark_19_angle_x",
    "Landmark_12_to_Landmark_13_angle_x",
    "Landmark_4_to_Landmark_16_distance",
    "Landmark_6_to_Landmark_11_distance",
    "Landmark_1_to_Landmark_8_angle_y",
    "Landmark_5_to_Landmark_10_distance",
    "Landmark_5_to_Landmark_10_angle_z",
    "Landmark_6_to_Landmark_16_angle_x",
    "Landmark_4_to_Landmark_8_angle_y",
    "Landmark_4_to_Landmark_11_distance",
    "Landmark_2_to_Landmark_12_angle_x",
    "Landmark_16_to_Landmark_17_angle_y",
    "Landmark_11_to_Landmark_19_distance",
    "Landmark_0_to_Landmark_1_angle_z",
    "Landmark_1_to_Landmark_10_angle_y",
    "Landmark_9_to_Landmark_13_angle_z",
    "Landmark_4_to_Landmark_5_angle_z",
    "Landmark_4_to_Landmark_8_angle_x",
    "Landmark_8_to_Landmark_13_angle_y",
    "Landmark_3_to_Landmark_5_angle_z",
    "Landmark_8_to_Landmark_9_distance",
    "Landmark_3_to_Landmark_11_distance",
    "Landmark_4_to_Landmark_15_distance",
    "Landmark_7_to_Landmark_11_distance",
    "pinky_direction_y",
    "Landmark_5_to_Landmark_11_distance",
    "Landmark_5_to_Landmark_12_angle_y",
    "Landmark_8_to_Landmark_18_distance",
    "Landmark_3_to_Landmark_4_angle_x",
    "Landmark_7_to_Landmark_9_angle_y",
    "Landmark_2_to_Landmark_20_distance",
    "Landmark_11_to_Landmark_12_angle_y",
    "Landmark_11_to_Landmark_15_angle_z",
    "Landmark_0_to_Landmark_14_angle_y",
    "Landmark_2_to_Landmark_3_angle_x",
    "Landmark_0_to_Landmark_4_angle_y",
    "Landmark_2_to_Landmark_16_angle_y",
    "Landmark_13_to_Landmark_16_angle_y",
    "Landmark_9_to_Landmark_11_angle_y",
    "Landmark_5_to_Landmark_7_angle_x",
    "Landmark_2_to_Landmark_11_angle_x",
    "Landmark_3_to_Landmark_9_angle_x",
    "Landmark_10_to_Landmark_19_angle_x",
    "Landmark_6_to_Landmark_13_distance",
    "ring_direction_x",
    "Landmark_4_to_Landmark_17_distance",
    "Landmark_9_to_Landmark_17_angle_z",
    "Landmark_3_to_Landmark_6_distance",
    "Landmark_14_to_Landmark_18_angle_y",
    "Landmark_3_to_Landmark_12_angle_y",
    "Landmark_13_to_Landmark_15_distance",
    "Landmark_1_to_Landmark_7_angle_z",
    "Landmark_15_to_Landmark_17_angle_x",
    "Landmark_13_to_Landmark_20_distance",
    "Landmark_10_to_Landmark_15_angle_z",
    "Landmark_2_to_Landmark_15_angle_x",
    "Landmark_5_to_Landmark_16_angle_y",
    "Landmark_3_to_Landmark_13_distance",
    "Landmark_0_to_Landmark_14_angle_z",
    "Landmark_2_to_Landmark_4_angle_x",
    "Landmark_1_to_Landmark_12_angle_x",
    "Landmark_5_to_Landmark_10_angle_y",
    "Landmark_13_to_Landmark_19_angle_x",
    "Landmark_6_to_Landmark_11_angle_x",
    "Landmark_0_to_Landmark_17_angle_y",
    "Landmark_18_to_Landmark_19_angle_y",
    "Landmark_4_to_Landmark_20_distance",
    "Landmark_1_to_Landmark_20_angle_z",
    "Landmark_2_to_Landmark_13_angle_y",
    "Landmark_2_to_Landmark_9_angle_z",
    "Landmark_5_to_Landmark_9_angle_x",
    "Landmark_7_to_Landmark_14_angle_x",
    "Landmark_0_to_Landmark_7_angle_y",
    "Landmark_5_to_Landmark_14_angle_z",
]
def record_landmarks(landmarks):
    """
    This function is responsible for collecting all features that will be feed into the kNN model to determine 
    each gesture. Note that this function also filters out the unimportant features that were recongized by 
    the Random Forest. Note that we haven't loaded the Random Forest model into this code, so we are just 
    manually removing them by iterating the features with the un_importatn features list
    """

    frame_data = {}

    # compute all features

    #distances
    distance_data = calculate_distance(landmarks)

    frame_data.update(distance_data)


    #angles

    angle_data = calculate_angles_between_pairs(landmarks)
    
    frame_data.update(angle_data)

    #directions

    direction_data = calculate_direction(landmarks)
    
    frame_data.update(direction_data)

    data.append(frame_data)  # Update global list if needed

    # filters the unimportant features out. 
    filtered_data = {key: value for key, value in frame_data.items() if key not in unimportant_features}

    return filtered_data

# update plot with dynamic axis limits
def update_plot(landmarks):

    if landmarks is None:
        return  


    #   I SCREWED UP THE PLOT ORIENTAITON SO PARDON THE WEIRD AXIS ORIENTATION
    x_data = [x for x, _, _ in landmarks]

    y_data = [-z for _, _, z in landmarks]
    
    z_data = [y for _, y, _ in landmarks]

    ax.set_xlim(min(x_data) - 0.1, max(x_data) + 0.1)

    ax.set_ylim(min(y_data) - 0.1, max(y_data) + 0.1)

    ax.set_zlim(min(z_data) - 0.1, max(z_data) + 0.1)

    scatter._offsets3d = (x_data, y_data, z_data)

    for i, (start, end) in enumerate(HAND_CONNECTIONS):

        line_x = [x_data[start], x_data[end]]

        line_y = [y_data[start], y_data[end]]

        line_z = [z_data[start], z_data[end]]

        lines[i].set_data(line_x, line_y)

        lines[i].set_3d_properties(line_z)

    fig.canvas.draw()  

    plt.pause(0.001)

def predict_gesture(feature_vector):
    """
    Predicts the gesture based on the processed feature vector using the kNN model.
    """
    predicted_label = knn_model.predict([feature_vector])[0]

    return gesture_mapping.get(predicted_label, "Unknown")


def main():

    global recording, calibrated, frame_count, data, label, label_entered

    cap = cv2.VideoCapture(0)

    calibrate_message_displayed = False

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:

        while cap.isOpened():
        
            success, image = cap.read()
        
            if not success:
        
                print("Ignoring empty camera frame.")
        
                continue

        
            image = cv2.flip(image, 1)
        
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
            results = hands.process(image_rgb)

        
            if results.multi_hand_landmarks:
        
                for hand_landmarks in results.multi_hand_landmarks:
        
                    if not calibrated:
        
                        if not calibrate_message_displayed:
        
                            print("Press 'c' to calibrate with an open palm.")
        
                            calibrate_message_displayed = True
        
                        continue



                    # normalize landmarks

                    normalized_landmark_data = normalize_landmarks(hand_landmarks.landmark)

                    # Recording features - angles, distances, and directions

                   # if normalized_landmark_data:
                   #     if recording:

                   #         record_landmarks(normalized_landmark_data)


                    if normalized_landmark_data:
                     # Process the normalized landmarks to extract 840 features
                        feature_data = record_landmarks(normalized_landmark_data)

                    

                        if feature_data:  # Ensure feature_data is not None

                            feature_vector = [value for value in feature_data.values()] # flattens the dictionary into a feature vector

                            print(f"Filtered Feature Vector Length: {len(feature_vector)}")

                            if len(feature_vector) == knn_model.n_features_in_:

                                gesture = predict_gesture(feature_vector) #Calls the predict_gesture function and prints the output

                                print(f"Predicted Gesture: {gesture}")

                            else:

                                 print(f"Feature vector size mismatch. Expected: {knn_model.n_features_in_}, Got: {len(feature_vector)}")

                        else:

                            print("Failed to extract features.")
                    



                    update_plot(normalized_landmark_data)

                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Hand Tracking", image)

            # controls
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and not calibrated:

                if results.multi_hand_landmarks:

                    calibrate_hand(results.multi_hand_landmarks[0].landmark)

                    calibrated = True

                    calibrate_message_displayed = False

            elif key == ord('q'):

                break


    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
