import cv2
import face_recognition
import os
import numpy as np
#68 face landmarks
unknown_dir = 'unknown'
known_dir = 'known'
known_name = []
name = []
known_encodings = []

def read_img(img):
    (h, w) = img.shape[:2]
    width = 500
    ratio = width / float(w)
    height = int(h * ratio)
    return cv2.resize(img, (width, height))

for file in os.listdir(known_dir):
    path = f'{known_dir}/{file}'
    img = cv2.imread(path)
    img = read_img(img)
    face_enc = face_recognition.face_encodings(img)[0]
    known_encodings.append(face_enc)
    known_name.append(file.split('.')[0])

cap = cv2.VideoCapture(0)

# Set the desired frame width and height
frame_width = 1280  # Replace with your desired width
frame_height = 720  # Replace with your desired height
cap.set(3, frame_width)
cap.set(4, frame_height)

while True:
    success, img = cap.read()
    img = read_img(img)
    face_enc = face_recognition.face_encodings(img)
    if len(face_enc) > 0:
        face_enc = face_enc[0]
        results = face_recognition.compare_faces(known_encodings, face_enc)
        distance = face_recognition.face_distance(known_encodings, face_enc)
        matchIndex = np.argmin(distance)
        for i, result in enumerate(results):
            if result:
                name = known_name[i]
                face_locations = face_recognition.face_locations(img)
                face_landmarks = face_recognition.face_landmarks(img)
                face_landmark = face_landmarks[0]
                top, right, bottom, left = face_locations[0]
                cv2.rectangle(img, (left, top), (right, bottom), color=(255, 0, 255), thickness=2)
                cv2.putText(img, name, (left - 20, top - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1)

                # # Draw facial landmarks
                for facial_feature, points in face_landmark.items():
                    for point in points:
                        cv2.circle(img, point, 2, (0, 0, 255), -1)
                # print(face_landmark.items());

    cv2.imshow("webcam", img)
    cv2.waitKey(1)
