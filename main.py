# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from frameextractor import frameExtractor
from handshape_feature_extractor import HandShapeFeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity


## import the handfeature extractor class

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

TRAIN_DATA_PATH = "traindata"
FRAMES_PATH = "trainframes"
OUTPUT_FILE = "train_vectors.npy"

TEST_DATA_PATH = "test"
TEST_FRAMES_PATH = "testframes"
TEST_OUTPUT_FILE = "test_vectors.npy"

gesture_to_label = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "DecreaseFanSpeed": 10,
    "FanOff": 11,
    "FanOn": 12,
    "IncreaseFanSpeed": 13,
    "LightOff": 14,
    "LightOn": 15,
    "SetThermo": 16
}

def clean_gesture_name(filename):
    base = os.path.splitext(filename)[0] 
    if base.startswith("H-"):
        base = base[2:]  
    return base


def generate_train_penultimate_layer():
    if not os.path.exists(FRAMES_PATH):
        os.mkdir(FRAMES_PATH)

    extractor = HandShapeFeatureExtractor.get_instance()
    feature_vectors = []
    count = 0

    for filename in os.listdir(TRAIN_DATA_PATH):
        if filename.endswith(".mp4"):
            count += 1
            video_path = os.path.join(TRAIN_DATA_PATH, filename)

            # Extract middle frame
            frameExtractor(video_path, FRAMES_PATH, count)
            frame_file = os.path.join(FRAMES_PATH, f"{count+1:05d}.png")


            if not os.path.exists(frame_file):
                continue

            # Load grayscale image
            img = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)

            if img is None or img.size == 0:
                continue

            try:
                vector = extractor.extract_feature(img)
                feature_vectors.append(vector.flatten())
            except Exception as e:
                continue

    feature_vectors = np.array(feature_vectors)
    np.save(OUTPUT_FILE, feature_vectors)

    return feature_vectors


# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video

def generate_test_penultimate_layer():
    if not os.path.exists(TEST_FRAMES_PATH):
        os.mkdir(TEST_FRAMES_PATH)

    extractor = HandShapeFeatureExtractor.get_instance()
    feature_vectors = []

    count = 0
    for filename in os.listdir(TEST_DATA_PATH):
        if filename.endswith(".mp4"):
            count += 1
            video_path = os.path.join(TEST_DATA_PATH, filename)

            # Extract and save the middle frame
            frameExtractor(video_path, TEST_FRAMES_PATH, count)
            frame_file = os.path.join(TEST_FRAMES_PATH, f"{count+1:05d}.png")

            if not os.path.exists(frame_file):
                continue

            img = cv2.imread(frame_file, cv2.IMREAD_GRAYSCALE)

            if img is None or img.size == 0:
                continue

            # Extract CNN features
            try:
                vector = extractor.extract_feature(img)
                feature_vectors.append(vector.flatten())
            except Exception as e:
                continue

    feature_vectors = np.array(feature_vectors)
    np.save(TEST_OUTPUT_FILE, feature_vectors)

    return feature_vectors


# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

def recognize_gestures(train_vectors, test_vectors, train_filenames):

    results = []

    for test_vec in test_vectors:
        sims = cosine_similarity([test_vec], train_vectors)[0]
        best_match_idx = np.argmax(sims)

        gesture_name = clean_gesture_name(train_filenames[best_match_idx])

        matched_label = None
        for key in gesture_to_label:
            if key.lower().replace(" ", "") in gesture_name.lower().replace(" ", ""):
                matched_label = gesture_to_label[key]
                break

        if matched_label is None:
            matched_label = -1

        results.append(matched_label)

    results = np.array(results)

    df = pd.DataFrame(results)
    df.to_csv("Results.csv", index=False, header=False)

    return results

if __name__ == "__main__":
    train_vectors = generate_train_penultimate_layer()
    test_vectors = generate_test_penultimate_layer()

    train_filenames = [f for f in os.listdir("traindata") if f.endswith(".mp4")]
    results = recognize_gestures(train_vectors, test_vectors, train_filenames)
