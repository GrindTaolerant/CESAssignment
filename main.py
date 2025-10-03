# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import tensorflow as tf
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor

def generate_training_penultimate_layer(training_videos_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    frames_dir = os.path.join(output_path, "extracted_frames")
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    feature_extractor = HandShapeFeatureExtractor.get_instance()
    training_features = []
    video_names = []
    video_files = []
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
     
    if os.path.exists(training_videos_path):
        for file in os.listdir(training_videos_path):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(file)

    for i, video_file in enumerate(video_files):
        video_path = os.path.join(training_videos_path, video_file)
         
        try:
            frameExtractor(video_path, frames_dir, i)
            frame_filename = f"{i+1:05d}.png"
            frame_path = os.path.join(frames_dir, frame_filename)
            
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                
                if frame is not None:   
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    features = feature_extractor.extract_feature(gray_frame)
                    flattened_features = features.flatten()
                    training_features.append(flattened_features)
                    video_names.append(video_file)
                    
        except Exception as e:
            continue
    
    if training_features:
        training_features_array = np.array(training_features)
        features_file = os.path.join(output_path, "training_penultimate_layer.npy")
        names_file = os.path.join(output_path, "training_video_names.npy")
        np.save(features_file, training_features_array)
        np.save(names_file, np.array(video_names))
        return training_features_array, video_names
    else:
        return None, None

def generate_test_penultimate_layer(test_videos_path, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    frames_dir = os.path.join(output_path, "extracted_test_frames")
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    feature_extractor = HandShapeFeatureExtractor.get_instance()
    test_features = []
    video_names = []
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    if os.path.exists(test_videos_path):
        for file in os.listdir(test_videos_path):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(file)
    
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(test_videos_path, video_file)
        
        try:
            frameExtractor(video_path, frames_dir, i)
            frame_filename = f"{i+1:05d}.png"
            frame_path = os.path.join(frames_dir, frame_filename)
            
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                
                if frame is not None:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    features = feature_extractor.extract_feature(gray_frame)
                    flattened_features = features.flatten()
                    test_features.append(flattened_features)
                    video_names.append(video_file)
                    
        except Exception as e:
            continue
    
    if test_features:
        test_features_array = np.array(test_features)
        features_file = os.path.join(output_path, "test_penultimate_layer.npy")
        names_file = os.path.join(output_path, "test_video_names.npy")
        np.save(features_file, test_features_array)
        np.save(names_file, np.array(video_names))
        return test_features_array, video_names
    else:
        return None, None

if __name__ == "__main__":
    training_videos_path = "traindata"
    output_path = "trainoutput"
    training_features, video_names = generate_training_penultimate_layer(training_videos_path, output_path)
    
    test_videos_path = "test"
    test_output_path = "testoutput"
    test_features, test_video_names = generate_test_penultimate_layer(test_videos_path, test_output_path)
    
    if training_features is not None and test_features is not None:
        results = gesture_recognition(test_features, test_video_names, training_features, video_names, test_output_path)

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

def gesture_recognition(test_features, test_video_names, training_features, training_video_names, output_path):
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)
    
    results = []
    
    for i, test_feature in enumerate(test_features):
        similarities = []
        
        for training_feature in training_features:
            similarity = cosine_similarity(test_feature, training_feature)
            similarities.append(similarity)
        
        max_similarity_index = np.argmax(similarities)
        results.append(max_similarity_index)
    
    with open("Results.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        for pred_label in results:
            writer.writerow([pred_label])
    
    return results

