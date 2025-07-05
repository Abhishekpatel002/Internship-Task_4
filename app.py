# app.py

import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from datetime import datetime
import os

# Load models
age_model = load_model('saved_models/age_model.h5')
gender_model = load_model('saved_models/gender_model.h5')

# Create CSV if not exists
if not os.path.exists("output.csv"):
    df = pd.DataFrame(columns=["Age", "Gender", "Time"])
    df.to_csv("output.csv", index=False)

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_input = preprocess_face(face_img)

        age = int(age_model.predict(face_input)[0][0])
        gender_pred = gender_model.predict(face_input)[0][0]
        gender = "Male" if gender_pred < 0.5 else "Female"

        label = f"{gender}, Age: {age}"

        if age > 60:
            label += " [Senior Citizen]"
            color = (0, 0, 255)  # Red
            # Save to CSV
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df = pd.read_csv("output.csv")
            df.loc[len(df.index)] = [age, gender, now]
            df.to_csv("output.csv", index=False)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Senior Citizen Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
