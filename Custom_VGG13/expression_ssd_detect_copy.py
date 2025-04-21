import streamlit as st
import lmstudio as lms
import cv2
import numpy as np
import os
from PIL import Image

# --- Emotion detection setup ---
def load_emotion_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(base_dir, 'emotion-ferplus-8.onnx')
    proto_path = os.path.join(base_dir, 'RFB-320', 'RFB-320.prototxt')
    model_path = os.path.join(base_dir, 'RFB-320', 'RFB-320.caffemodel')
    missing = [p for p in [onnx_path, proto_path, model_path] if not os.path.exists(p)]
    if missing:
        st.error(f"No se encuentran los archivos de modelos: {', '.join(missing)}")
        st.stop()
    emotion_net = cv2.dnn.readNetFromONNX(onnx_path)
    face_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    return emotion_net, face_net

emotion_dict = {
    0: 'neutral', 1: 'happiness', 2: 'surprise',
    3: 'sadness', 4: 'anger', 5: 'disgust', 6: 'fear'
}

# Precompute SSD priors
from math import ceil
image_std = 128.0
strides = [8.0, 16.0, 32.0, 64.0]
min_boxes = [[10.0,16.0,24.0],[32.0,48.0],[64.0,96.0],[128.0,192.0,256.0]]

def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes):
    priors = []
    for idx in range(len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][idx]
        scale_h = image_size[1] / shrinkage_list[1][idx]
        for j in range(feature_map_list[1][idx]):
            for i in range(feature_map_list[0][idx]):
                x_center = (i + .5) / scale_w
                y_center = (j + .5) / scale_h
                for mb in min_boxes[idx]:
                    w_ = mb / image_size[0]
                    h_ = mb / image_size[1]
                    priors.append([x_center, y_center, w_, h_])
    return np.clip(priors, 0.0, 1.0)

def define_priors(image_size):
    fmap_sizes = [[ceil(image_size[0]/s) for s in strides],
                  [ceil(image_size[1]/s) for s in strides]]
    shrink = [strides, strides]
    return generate_priors(fmap_sizes, shrink, image_size, min_boxes)

priors = define_priors([320,240])

# Emotion detection using SSD with correct aspect ratio resizing
# Emotion detection using SSD with debugging improvements
def detect_emotion_ssd(img_pil, emotion_net, face_net, priors, conf_threshold=0.3):
    # Convert and inspect frame
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    st.write("Frame shape (h, w):", (h, w))

    # Resize for SSD with correct aspect ratio
    target_size = (320, 240)
    aspect_ratio = w / h

    if aspect_ratio > 1:
        # Wide image: adjust width and pad height
        new_w = target_size[0]
        new_h = int(target_size[0] / aspect_ratio)
        pad_top = (target_size[1] - new_h) // 2
        pad_bottom = target_size[1] - new_h - pad_top
        pad_left, pad_right = 0, 0  # No padding needed for width
    else:
        # Tall or square image: adjust height and pad width
        new_h = target_size[1]
        new_w = int(target_size[1] * aspect_ratio)
        pad_left = (target_size[0] - new_w) // 2
        pad_right = target_size[0] - new_w - pad_left
        pad_top, pad_bottom = 0, 0  # No padding needed for height

    resized = cv2.resize(frame, (new_w, new_h))
    padded_frame = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(127, 127, 127))
    st.write("Resized and padded frame shape:", padded_frame.shape)

    # Create blob
    blob = cv2.dnn.blobFromImage(padded_frame, 1/image_std, target_size, 127)
    st.write("Blob shape:", blob.shape)
    face_net.setInput(blob)
    boxes, scores = face_net.forward(['boxes', 'scores'])
    boxes = np.reshape(boxes, (-1, 4))
    scores = np.reshape(scores, (-1, 2))
    st.write("Raw boxes:", boxes[:5])  # Debugging first 5 boxes
    st.write("Raw scores (face vs background):", scores[:5, 1], "... total", len(scores))

    # Filter detections based on confidence
    valid = [(i, sc[1]) for i, sc in enumerate(scores) if sc[1] > conf_threshold]
    if not valid:
        st.warning(f"SSD did not detect any faces with confidence > {conf_threshold}")
        return None

    # Best detection
    best_idx = max(valid, key=lambda x: x[1])[0]
    box = boxes[best_idx]
    x1, y1, x2, y2 = (box * [w, w, h, h]).astype(int)
    st.write(f"Best bounding box: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")

    # Check for invalid bounding box
    if x2 <= x1 or y2 <= y1:
        st.warning("Invalid SSD box: empty ROI.")
        return None

    # Draw the selected box
    debug_sel = frame.copy()
    cv2.rectangle(debug_sel, (x1, y1), (x2, y2), (0, 255, 0), 2)
    st.image(debug_sel, caption=f"Best SSD box (score={scores[best_idx, 1]:.2f})", use_container_width=True)

    # Crop face & detect emotion
    face = frame[y1:y2, x1:x2]
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (64, 64))
    blob2 = cv2.dnn.blobFromImage(gray_face, scalefactor=1 / 255.0)
    emotion_net.setInput(blob2)
    out = emotion_net.forward()
    label = int(np.argmax(out[0]))
    return emotion_dict.get(label)

# --- Streamlit app ---
st.set_page_config(page_title="Emotional Assistance")
st.title("🗣️ Emotional Assistance Chatbot")

# Load models
if 'emotion_models' not in st.session_state:
    st.session_state.emotion_net, st.session_state.face_net = load_emotion_models()

# Camera capture and emotion detection
def capture_and_detect():
    img_file = st.camera_input("Capture your face for emotion detection")
    if not img_file:
        return None
    img = Image.open(img_file)
    st.image(img, caption="Input from camera", use_container_width=True)
    return detect_emotion_ssd(img,
                               st.session_state.emotion_net,
                               st.session_state.face_net,
                               priors,
                               conf_threshold=0.3)

emotion = capture_and_detect()
if emotion:
    st.success(f"Detected emotion: {emotion}")

# Initialize chatbot model
if 'model' not in st.session_state:
    st.session_state.model = lms.llm("llama-3.2-1b-instruct")

# Build initial context
if 'full_context' not in st.session_state:
    base = "You are a professional therapist specializing in mental health."
    if emotion:
        base += f" The user appears to be feeling {emotion}."
    base += " You respond with empathy, validate emotions, and offer guidance non-judgmentally."
    st.session_state.full_context = [{"role":"system","content":base}]
    st.session_state.messages = []

# Chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if user_input := st.chat_input("Say something..."):
    st.session_state.full_context.append({"role":"user","content":user_input})
    st.session_state.messages.append({"role":"user","content":user_input})
    conv = ''
    for m in st.session_state.full_context:
        prefix = 'User:' if m['role']=='user' else 'Therapist:'
        conv += f"{prefix} {m['content']}\n"
    conv += "Therapist:"
    resp = st.session_state.model.respond(conv)
    st.session_state.messages.append({"role":"assistant","content":resp})
    with st.chat_message("assistant"):
        st.markdown(resp)
