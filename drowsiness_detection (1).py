import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from pygame import mixer
import time

# Initialize pygame mixer for alarm
mixer.init()
mixer.music.load("alarm.wav")  # Place an alarm.wav file in the same directory

# Eye Aspect Ratio threshold and consecutive frame count
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 20

# Load dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Facial landmark indices for left and right eye
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

def eye_aspect_ratio(eye_points):
    """Calculate Eye Aspect Ratio (EAR)"""
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def get_eye_points(landmarks, indices):
    return [(landmarks.part(i).x, landmarks.part(i).y) for i in indices]

def main():
    cap = cv2.VideoCapture(0)
    frame_counter = 0
    alarm_on = False

    print("Driver Drowsiness Detection Started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            left_eye = get_eye_points(landmarks, LEFT_EYE)
            right_eye = get_eye_points(landmarks, RIGHT_EYE)

            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # Draw eye contours
            left_hull = cv2.convexHull(np.array(left_eye))
            right_hull = cv2.convexHull(np.array(right_eye))
            cv2.drawContours(frame, [left_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_hull], -1, (0, 255, 0), 1)

            if avg_EAR < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CONSECUTIVE_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    if not alarm_on:
                        mixer.music.play(-1)
                        alarm_on = True
            else:
                frame_counter = 0
                if alarm_on:
                    mixer.music.stop()
                    alarm_on = False

            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    mixer.music.stop()

if __name__ == "__main__":
    main()
