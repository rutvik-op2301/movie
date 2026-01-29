import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Lazy globals
movies = None
tfidf = None
tfidf_matrix = None
indices = None

def load_model():
    global movies, tfidf, tfidf_matrix, indices

    if movies is not None:
        return

    movies = pd.read_csv("data/movies.csv")
    movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

    tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    indices = pd.Series(movies.index, index=movies["title"])

def recommend_movies(movie_title: str, top_n: int = 5):
    load_model()

    if movie_title not in indices:
        return []

    idx = indices[movie_title]

    # Compute similarity ONLY for one movie (not full matrix)
    sim_scores = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    similar_indices = sim_scores.argsort()[::-1][1:top_n + 1]
    return movies["title"].iloc[similar_indices].tolist()
