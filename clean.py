import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):

        image_name = os.path.join(DATA_DIR, dir_, img_path)
        image_number = int(os.path.splitext(os.path.basename(image_name))[0])
        if image_number >= 100:
            # Delete the image
            os.remove(image_name)
            print(f"Deleted: {image_name}")
