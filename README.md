Project Overview

This project focuses on music analysis, genre classification, and song recommendation using machine learning techniques. The project leverages the GTZAN dataset to explore different musical genres and improve recommendation systems. We used various models and data processing techniques to classify music genres and provide recommendations based on song similarity.

1.Data

GTZAN Dataset: This dataset contains 1000 audio files (30 seconds each) from 10 different music genres, such as classical, metal, jazz, and pop.

Processed Data: The dataset was processed to extract relevant audio features using Librosa, such as:

Mel-frequency cepstral coefficients (MFCCs)

Spectral Centroids

Chroma Frequencies

Zero Crossing Rate

Mel Spectrograms: Audio files were converted into mel spectrograms, allowing us to leverage image-based techniques like CNNs in future work.

2.Preprocessing

Silence Removal: Silence was removed from the beginning and end of audio tracks to focus only on the core music data.

Feature Scaling: Features were scaled using Min-Max scaling to ensure uniform input for machine learning models.

Handling Missing Data: Missing values in the dataset were imputed to maintain data integrity.

3.Exploratory Data Analysis (EDA)

Visualization of Temporal Features: Analyzed the rhythm, tempo, and beats of the audio tracks. For example, genres like metal showed dense spectral content, while classical displayed a wide dynamic range.

BPM Analysis: Box plots were created to compare beats per minute (BPM) across different genres. This was particularly helpful in feature engineering for genre classification models.

Principal Component Analysis (PCA): PCA was used to reduce the dimensionality of the feature space and visualize the genre clusters. This helped in understanding how well the genres are separated in the feature space.

4.Model Training and Evaluation

Several models were trained and evaluated, including:

K-Nearest Neighbors (KNN): Achieved the highest accuracy of 92% after hyperparameter tuning with the Manhattan distance metric and distance weighting.

XGBoost: Grid search was used to fine-tune this model, achieving an accuracy of 90.79%.

Random Forest: Also performed well but had lower accuracy compared to KNN.

Model Performance: Precision, recall, and F1-scores were calculated for all models, with KNN showing balanced performance across all genres.

4.Recommendation System

Cosine Similarity: The recommendation system compares audio tracks based on features like pitch, tempo, and timbre. Cosine similarity was used to recommend songs that are most similar to the user’s query.

How to Run Project
Clone The repo and install all dependacies

Run Notebooks You can explore each stage of the project using the provided Jupyter notebooks. Start with the EDA notebook to understand the data and then proceed with model training and evaluation.

Use the Recommendation System You can use the recommendation.py script to generate song recommendations. Provide a song file as input, and it will return the most similar songs based on audio features.

References
Ma, J. (2023, April 20). Music Genres Classification and Recommendation System. Medium.
Elbir, A., & Aydın, N. (2020). Music genre classification and music recommendation using deep learning.
Gupta, A., & Kumar, G. (2023). Music Genre Classification and Recommendation System.
Kaggle: Work with Audio Data (Visualize, Classify, Recommend)
TensorFlow Datasets: GTZAN
