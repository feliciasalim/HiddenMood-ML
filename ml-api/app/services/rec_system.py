import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, "models", "recommender_system", "video_recommendation_dataset.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "recommender_system", "content_features.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "recommender_system", "vectorizer.pkl")

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = joblib.load(f)
with open(FEATURES_PATH, "rb") as f:
    content_features = joblib.load(f)

video_df = pd.read_csv(DATASET_PATH)

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
