"""
Emotion Detection:
Model from: https://github.com/onnx/models/blob/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx
Model name: emotion-ferplus-8.onnx
"""

import cv2
import numpy as np
import time
import os

from cv2 import dnn
from math import ceil

image_mean = np.array([127, 127, 127])  # Mean values for image normalization across RGB channels.
image_std = 128.0   # Standard deviation for image normalization.
iou_threshold = 0.3 # Threshold for Intersection over Union (IoU) metric to determine bounding box matches.
center_variance = 0.1   # Scaling factor for predicted bounding box center coordinates.
size_variance = 0.2 # Scaling factor for predicted bounding box dimensions.
min_boxes = [
    [10.0, 16.0, 24.0], 
    [32.0, 48.0], 
    [64.0, 96.0], 
    [128.0, 192.0, 256.0]
]   # Minimum bounding box dimensions for objects of different sizes.
strides = [8.0, 16.0, 32.0, 64.0]   # Control the scale of the feature maps according to the image size. Determines how much the filter shifts after each operation.
threshold = 0.5 # Confidence threshold for object detection.

def define_img_size(image_size):
    '''
    Calculate feature map dimensions based on the provided image size and a set of predefined stride values.
    These feature map dimensions reflect the expected output dimensions of CNN layers for varying scales of the input image.
    The priors in SSD provide an efficient way to simultaneously predict multiple bounding boxes and their associated class
    scores in a single forward pass of the network, enabling real-time object detection.
    '''
    shrinkage_list = [] # replicates the `strides` list for each element in the `image_size` list
    feature_map_w_h_list = []   # Calculate feature map dimensions based on the provided image size and a set of predefined stride values.
    for size in image_size:
        feature_map = [int(ceil(size / stride)) for stride in strides]  # ceil rounds a decimal number up to the nearest integer.
        feature_map_w_h_list.append(feature_map)

    for _ in range(0, len(image_size)):
        shrinkage_list.append(strides)

    priors = generate_priors(
        feature_map_w_h_list, shrinkage_list, image_size, min_boxes
    )
    return priors   # predicted the multiple bounding boxes and their class scores in a single forward pass of the network


def generate_priors(
    feature_map_list, shrinkage_list, image_size, min_boxes
):
    '''
    Utilize the feaure map, shrinkage list, image size and min boxes to create and return the desired prior bounding boxes.
    '''
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    print("priors nums:{}".format(len(priors)))
    return np.clip(priors, 0.0, 1.0)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    '''
    Hard Non-Maximum Suppression processes `box_scores` with parameters like `iou_threshold`, `top_k`, and `candidate_size`.
    It selects high-scoring, non-overlapping boxes through looping and IoU computation.
    '''
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)    # Order the scores from the bigger to lower
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]    # refined subset of boxes, improving object detection accuracy.


def area_of(left_top, right_bottom):
    '''
    Given the left top and the right bototn, it calculate the area of the bounding box
    It subtracts the top-left coordinates from the bottom-right coordinates to calculate the width and height of the rectangle ensuring that aren't negative.'''
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    '''
    IOU is the ratio of the area of overlap between two bounding boxes to the area of their union.
    Mathematically is the Area Of Intersection divided by the area of union.
    boxes0 and boxes1 are the bounding boxes and eps is a  small value added to the denominator to avoid division by zero.
    '''
    # Calculate the coordinates of the overlaping region, the intersection of the two boxes
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2]) # x_mas and y_max of the overlaping region
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:]) 

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def predict(
    width, 
    height, 
    confidences, 
    boxes, 
    prob_threshold, 
    iou_threshold=0.3, 
    top_k=-1
):
    # Extract the first batch of boxes and confidences
    boxes = boxes[0]
    confidences = confidences[0]

    # Lists to store the final selected boxes and their corresponding labels
    picked_box_probs = []
    picked_labels = []

    # Iterate over each class (starting from 1, skipping the background class)
    for class_index in range(1, confidences.shape[1]):
        # Extract the confidence scores for the current class
        probs = confidences[:, class_index]

        # Create a mask for boxes with confidence scores above the threshold
        mask = probs > prob_threshold
        probs = probs[mask]

        # Skip if no boxes meet the confidence threshold
        if probs.shape[0] == 0:
            continue

        # Select the subset of boxes that meet the confidence threshold
        subset_boxes = boxes[mask, :]

        # Concatenate the boxes and their corresponding confidence scores
        box_probs = np.concatenate(
            [subset_boxes, probs.reshape(-1, 1)], axis=1
        )

        # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
        box_probs = hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             )
        
        # Append the filtered boxes and their labels
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    
    # If no boxes were picked, return empty arrays
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    
    # Concatenate all picked boxes into a single array
    picked_box_probs = np.concatenate(picked_box_probs)

    # Scale the box coordinates back to the original image dimensions
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height

    # Return the final boxes, labels, and confidence scores
    return (
        picked_box_probs[:, :4].astype(np.int32),   # Bounding box coordinates
        np.array(picked_labels),    # Class labels
        picked_box_probs[:, 4]  # Confidence scores
    )


def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    # Convert predicted offsets to bounding box coordinates based on prior boxes
    if len(priors.shape) + 1 == len(locations.shape):
        # Expand dimensions of priors if necessary for broadcasting
        priors = np.expand_dims(priors, 0)
    return np.concatenate([
        # Calculate center coordinates (cx, cy)
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        # Calculate width and height (w, h)
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=len(locations.shape) - 1)


def center_form_to_corner_form(locations):
    # Convert bounding box coordinates from center form (cx, cy, w, h)
    # to corner form (x_min, y_min, x_max, y_max)
    return np.concatenate(
        [
            locations[..., :2] - locations[..., 2:] / 2,  # Top-left corner
            locations[..., :2] + locations[..., 2:] / 2   # Bottom-right corner
        ], 
        len(locations.shape) - 1
    )


def FER_live_cam():
    emotion_dict = {
        0: 'neutral', 
        1: 'happiness', 
        2: 'surprise', 
        3: 'sadness',
        4: 'anger', 
        5: 'disgust', 
        6: 'fear'
    }

    # Open webcam (or replace with a video file path)
    cap = cv2.VideoCapture(0)

    # Get frame dimensions
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    # Initialize video writer to save the output
    result = cv2.VideoWriter('infer2-test.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

    # Load ONNX model for emotion detection
    model = cv2.dnn.readNetFromONNX('/Users/bernardoquindimil/Code/Berniquindimil/Proyect/Custom_VGG13/emotion-ferplus-8.onnx')
    
    # Load Caffe face detector
    model_path = '/Users/bernardoquindimil/Code/Berniquindimil/Proyect/Custom_VGG13/RFB-320/RFB-320.caffemodel'
    proto_path = '/Users/bernardoquindimil/Code/Berniquindimil/Proyect/Custom_VGG13/RFB-320/RFB-320.prototxt'
    net = dnn.readNetFromCaffe(proto_path, model_path)

    # Define input size for the face detector
    input_size = [320, 240]
    width = input_size[0]
    height = input_size[1]

    # Generate prior boxes for object detection
    priors = define_img_size(input_size)

    emotion_count = {emotion: 0 for emotion in emotion_dict.values()}  # Initialize emotion counters
    iteration_count = 0
    emotion_history = []  # To keep track of the most frequent emotions in batches
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_ori = frame

            # Resize and preprocess the frame for the face detector
            rect = cv2.resize(img_ori, (width, height))
            rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
            net.setInput(dnn.blobFromImage(
                rect, 1 / image_std, (width, height), 127)
            )

            # Perform face detection
            start_time = time.time()
            boxes, scores = net.forward(["boxes", "scores"])
            boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
            scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)

            # Convert box predictions to corner form
            boxes = convert_locations_to_boxes(
                boxes, priors, center_variance, size_variance
            )
            boxes = center_form_to_corner_form(boxes)

            # Perform Non-Maximum Suppression (NMS) and filter predictions
            boxes, labels, probs = predict(
                img_ori.shape[1], 
                img_ori.shape[0], 
                scores, 
                boxes, 
                threshold
            )

            # Convert frame to grayscale for emotion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Iterate over detected boxes
            for (x1, y1, x2, y2) in boxes:
                # Calculate width and height of the bounding box
                w = x2 - x1
                h = y2 - y1

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Extract and preprocess the face region for emotion detection
                resize_frame = cv2.resize(
                    gray[y1:y1 + h, x1:x1 + w], (64, 64)
                )
                resize_frame = resize_frame.reshape(1, 1, 64, 64)

                # Perform emotion detection
                model.setInput(resize_frame)
                output = model.forward()

                # Get the predicted emotion label
                pred = emotion_dict[list(output[0]).index(max(output[0]))]

                # Update emotion count
                emotion_count[pred] += 1
                emotion_history.append(pred)
                
                # Draw the emotion label on the frame
                cv2.rectangle(
                    img_ori, 
                    (x1, y1), 
                    (x2, y2), 
                    (215, 5, 247), 
                    2,
                    lineType=cv2.LINE_AA
                )
                cv2.putText(
                    frame, 
                    pred, 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (215, 5, 247), 
                    2,
                    lineType=cv2.LINE_AA
                )

            # Write the processed frame to the output video
            result.write(frame)
        
            # Display the frame in a window
            cv2.imshow('frame', frame)

            # Increment iteration count
            iteration_count += 1

            # Every 30 frames (or however many iterations you prefer), compute the most frequent emotion
            if iteration_count % 30 == 0:
                most_common_emotion = max(emotion_count, key=emotion_count.get)
                print(f"Most common emotion in last 30 frames: {most_common_emotion}")
                emotion_count = {emotion: 0 for emotion in emotion_dict.values()}  # Reset counters for the next batch
                
                # Here you can call another function, passing the most_common_emotion as input
                # Example: your_function(most_common_emotion)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release resources
    cap.release()
    result.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    FER_live_cam()
    