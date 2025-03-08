import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

# Load the model architecture from a JSON file (if you have it)
with open("path_to_your_model.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)  # Recreate the model

# Load the weights
model.load_weights("path_to_your_model_weights.h5")

# Compile the model (necessary for predictions)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels from FER-2013 dataset
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Extract face
        face = cv2.resize(face, (48, 48))  # Resize to 48x48 (FER-2013 size)
        face = face / 255.0  # Normalize pixel values
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = np.expand_dims(face, axis=-1)  # Add channel dimension

        # Predict emotion
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        # Draw rectangle around face and label it
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
