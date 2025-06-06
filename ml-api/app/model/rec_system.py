import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

BASE_DIR = "models"
DATASET_PATH = "video_recommendation_dataset.pkl"
FEATURES_PATH = "content_features.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    content_features = pickle.load(f)

video_df = pd.read_pickle(DATASET_PATH)

def recommend_video(input_text, top_n=2, similarity_threshold=0.2):
    input_vector = vectorizer.transform([input_text])
    similarity_scores = cosine_similarity(input_vector, content_features).flatten()

    filtered_indices = np.where(similarity_scores >= similarity_threshold)[0]

    if len(filtered_indices) == 0:
        return []

    sorted_indices = filtered_indices[np.argsort(similarity_scores[filtered_indices])[::-1]]
    final_indices = sorted_indices[:top_n]

    results = video_df.iloc[final_indices][['Link']].copy()

    return results['Link'].tolist()
