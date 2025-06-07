# HiddenMood FastAPI Documentation

HiddenMood FastAPI is a mental health backend service designed to help individuals gain deeper insight into their emotional well-being through the power of machine learning and natural language processing. The primary objective of this system is to analyze user-submitted text to automatically classify both stress levels and emotions, offering users a data-driven reflection of their mental condition.

This model works by using a Bidirectional Long Short-Term Memory (BiLSTM) model trained to detect two key mental health indicators: stress levels, which are categorized as Low, Medium, or High; and emotional states, like Anxious, Depressed, Lonely, Overwhelmed, and Panicked. This dual classification helps create a nuanced picture of a user’s mental status based on the linguistic patterns and expressions in their input.

Based on the detected emotions, there's a video recommender system which recommends relevant videos aimed at emotional regulation and relaxation, serving as immediate self-help resources. Furthermore, HiddenMood employs Google’s Vertex AI to deliver thoughtful, AI-generated feedback and coping strategies tailored to the user’s current mental state.

---


## Table of Contents

* [Features](#features)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Installation (Local)](#installation-local)
* [Deployment via GCP](#deployment-via-gcp)
* [License](#license)

---


## Features

* Text analysis using BiLSTM for stress and emotion classification
* Stress level output as percentage (0–100%)
* Emotion-based video recommendation system
* Generative feedback and coping strategies using Vertex AI

---


## Dataset

To train the HiddenMood classification model, the mental health dataset was taken from two key sources:

### 1. Reddit Mental Health Posts (Scraped via PRAW)

  Using the Python Reddit API Wrapper (PRAW), approximately 10,000 posts were scraped from various mental health subreddits, including:
  
  * r/depression
  * r/anxiety
  * r/mentalhealth
  * r/stress

### 2. Kaggle Dataset: Reddit Mental Health Data by Neel Ghoshal

  To add more scraped data and improve the model's generalizability, we used an open-source dataset from Kaggle:
  
  * Title: Reddit Mental Health Data
  * Author: Neel Ghoshal
  * Link: [https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data](https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data)

---


## Model Architecture

* Sentiment Model: BiLSTM (Bidirectional LSTM)
* Emotion-based Recommender: TF-IDF + Cosine Similarity (Content-Based Filtering)
* Feedback System: Vertex AI 

---


## Installation (Local)

To install this API, follow these steps:

  1. Clone repository
  
      ```
      git clone https://github.com/your-username/HiddenMood-ML-API.git
      cd HiddenMood-ml-api
      ```
  
  2. Create virtual environment
  
      ```
      python -m venv venv
      source venv/bin/activate  # On Windows use: venv\Scripts\activate
      ```
  
  3. Install dependencies
  
      ```
      pip install -r requirements.txt
      ```
  
  4. Run API
  
      ```
      uvicorn main:app --reload
      ```
      
  5. Testing via [http://localhost:8000/docs](http://localhost:8000/docs)

---


## Deployment via GCP

Ensure you have Docker installed and your GCP project + Secret Manager are set up. Then follow these steps:

1. Clone repository

    ```
    git clone https://github.com/your-username/HiddenMood-ML-API.git
    cd HiddenMood-ml-api
    ```

2. Deploy to Docker and GCP

    ```
    docker build -t gcr.io/[APP-ID]/ml-api .
    docker push gcr.io/[APP-ID]/ml-api
    gcloud run deploy ml-api \
      --image gcr.io/[APP-ID]/ml-api \
      --region asia-southeast1 \
      --service-account ml-api-service-account@[APP-ID].iam.gserviceaccount.com \
      --memory 1Gi \
      --allow-unauthenticated
    ```

3. Testing
   
    ```
    curl -X POST "https://your-cloud-run-url/predict/analyze" \
      -H "Content-Type: application/json" \
      -d '{"text": "I am feeling stressed"}'
    ```

---


## License

This project is licensed under the [MIT License](./LICENSE).
