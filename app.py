from mtcnn import MTCNN
import cv2
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model

model = load_model("mask_detector_model.h5") 

class_map = {'with_mask': 0, 'without_mask': 1}
inv_map = {v: k for k, v in class_map.items()}

detector = MTCNN()
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_face(img):
    """Try multiple face detection methods"""
                                            
    # 1. Try MTCNN first
    faces = detector.detect_faces(img)
    if faces:
        x, y, width, height = faces[0]['box']
        return max(0, x), max(0, y), width, height

    # 2. If MTCNN fails, try Haar Cascade
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces_haar = haar_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=4, 
        minSize=(30, 30)
    )
    if len(faces_haar) > 0:
        x, y, width, height = faces_haar[0]
        return x, y, width, height

    # 3. If both fail, try with the whole image
    return 0, 0, img.shape[1], img.shape[0]

def predict_mask(img):
    # Convert image to numpy array if needed
    img = np.array(img)
    
    # Handle grayscale images
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Get face coordinates
    x, y, width, height = detect_face(img)
    
    # Crop and preprocess the face
    face = img[y:y+height, x:x+width]
    face = cv2.resize(face, (224, 224))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    
    # Make prediction
    pred = model.predict(face, verbose=0)[0][0]
    class_idx = int(pred > 0.5)
    label = inv_map[class_idx]
    
    # Calculate confidence percentage
    confidence = pred if class_idx == 1 else 1 - pred
    confidence = round(confidence * 100, 2)
    
    # Add emoji based on prediction
    emoji = "😷" if "With" in label else "😐"
    
    return f"{label} ({confidence}% confident) {emoji}"

gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Face Mask Detector",
    description="Upload a face image to detect if a mask is present."
).launch()
