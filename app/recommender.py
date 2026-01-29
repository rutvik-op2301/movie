import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data
movies = pd.read_csv("data/movies.csv")

# Clean genres
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

# Vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Index mapping
indices = pd.Series(movies.index, index=movies["title"])

def recommend_movies(movie_title: str, top_n: int = 5):
    if movie_title not in indices:
        return []

    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    return movies["title"].iloc[[i[0] for i in sim_scores]].tolist()
