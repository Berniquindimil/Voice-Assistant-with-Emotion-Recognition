import cv2
import numpy as np
import time
import os

# Define emotion labels
emotion_dict = {
    0: 'neutral',
    1: 'happiness',
    2: 'surprise',
    3: 'sadness',
    4: 'anger',
    5: 'disgust',
    6: 'fear'
}

def load_emotion_model():
    """
    Load the emotion recognition model, falling back to a simpler method if ONNX fails
    """
    try:
        # Try to load the ONNX model
        emotion_model_path = 'Custom_VGG13/emotion-ferplus-8.onnx'
        if os.path.exists(emotion_model_path):
            print(f"Loading emotion model from {emotion_model_path}")
            return cv2.dnn.readNetFromONNX(emotion_model_path), "onnx"
        else:
            print(f"Emotion model file not found at {emotion_model_path}")
            # As a fallback, check for other model formats
            if os.path.exists('emotion_model.h5'):
                print("Using Keras model as fallback")
                from keras.models import load_model
                return load_model('emotion_model.h5'), "keras"
            else:
                print("No emotion model found. Using a placeholder for demonstration.")
                return None, "placeholder"
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        print("Using placeholder for demonstration")
        return None, "placeholder"

def process_face_haar(frame):
    """
    Detect faces using Haar Cascade classifier (more stable than DNN)
    """
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces, gray

def predict_emotion(face_roi, model, model_type):
    """
    Predict emotion from a face region of interest
    """
    if model_type == "placeholder":
        # For demonstration, return a random emotion if no model is available
        import random
        emotion_idx = random.randint(0, 6)
        return emotion_dict[emotion_idx], 0.7
    
    elif model_type == "onnx":
        # Preprocess face for ONNX model
        try:
            # Resize to 64x64 which is common for FER models
            resized_face = cv2.resize(face_roi, (64, 64))
            
            # Normalize and reshape for the model
            processed_face = resized_face.reshape(1, 1, 64, 64)
            
            # Run inference
            model.setInput(processed_face)
            output = model.forward()
            
            # Get the predicted emotion
            emotion_idx = np.argmax(output[0])
            confidence = float(output[0][emotion_idx])
            
            return emotion_dict.get(emotion_idx, "unknown"), confidence
        except Exception as e:
            print(f"Error in emotion prediction: {e}")
            return "unknown", 0.0
    
    elif model_type == "keras":
        # Preprocess face for Keras model
        try:
            # Resize to 48x48 which is common for FER datasets
            resized_face = cv2.resize(face_roi, (48, 48))
            
            # Normalize and reshape for the model
            processed_face = resized_face / 255.0
            processed_face = np.expand_dims(processed_face, axis=0)
            processed_face = np.expand_dims(processed_face, axis=-1)  # Add channel dimension
            
            # Run inference
            prediction = model.predict(processed_face)[0]
            
            # Get the predicted emotion
            emotion_idx = np.argmax(prediction)
            confidence = float(prediction[emotion_idx])
            
            return emotion_dict.get(emotion_idx, "unknown"), confidence
        except Exception as e:
            print(f"Error in emotion prediction: {e}")
            return "unknown", 0.0
    
    return "unknown", 0.0

def run_emotion_detection():
    """
    Main function to run the emotion detection system with webcam
    """
    # Load the emotion model
    emotion_model, model_type = load_emotion_model()
    print(f"Using model type: {model_type}")
    
    # Initialize webcam
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
    except Exception as e:
        print(f"Error initializing webcam: {e}")
        return
    
    # Get frame dimensions
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    # Initialize video writer
    try:
        output_video = cv2.VideoWriter(
            'emotion_detection_result.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            10,
            (frame_width, frame_height)
        )
    except Exception as e:
        print(f"Error creating video writer: {e}")
        output_video = None
    
    print("Starting emotion recognition. Press 'q' to quit.")
    
    # Performance tracking
    frame_count = 0
    fps_list = []
    start_time_total = time.time()
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            start_time = time.time()
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Detect faces using Haar cascade (more stable than DNN)
            faces, gray = process_face_haar(frame)
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract the face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Skip if face ROI is empty
                if face_roi.size == 0:
                    continue
                
                # Predict emotion
                emotion, confidence = predict_emotion(face_roi, emotion_model, model_type)
                
                # Display the emotion label
                label = f"{emotion} ({confidence:.2f})"
                cv2.putText(
                    display_frame,
                    label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
            
            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            fps_list.append(fps)
            
            # Display FPS on frame
            avg_fps = sum(fps_list[-30:]) / min(len(fps_list), 30)
            cv2.putText(
                display_frame,
                f"FPS: {avg_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            
            # Write frame to output video
            if output_video is not None:
                output_video.write(display_frame)
            
            # Display frame
            cv2.imshow('Emotion Recognition', display_frame)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Calculate overall statistics
    total_time = time.time() - start_time_total
    avg_fps_overall = frame_count / total_time if total_time > 0 else 0
    
    print(f"\nProcessing complete:")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {avg_fps_overall:.2f}")
    
    # Release resources
    cap.release()
    if output_video is not None:
        output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_emotion_detection()