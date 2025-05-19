import cv2
import numpy as np

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")
age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")

# Labels for gender and age
gender_list = ['Male', 'Female']  
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Input dimensions for both models
input_width = 227
input_height = 227

# Function to predict gender and age
def predict(face_image):
    # Prepare the image for prediction
    blob = cv2.dnn.blobFromImage(face_image, 1.0, (input_width, input_height), (104.0, 177.0, 123.0), swapRB=False)

    # Predict gender
    gender_net.setInput(blob)
    gender = gender_list[np.argmax(gender_net.forward())]

    # Predict age
    age_net.setInput(blob)
    age = age_list[np.argmax(age_net.forward())]

    return gender, age

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_image = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_image, (input_width, input_height))

        # Predict gender and age
        gender, age = predict(face_resized)

        # Draw rectangle and display text
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"Gender: {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {age}", (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Gender and Age Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  