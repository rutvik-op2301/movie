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

def recommend_movies(movie_name, n_recommendations=5):
    movie_name = movie_name.lower()

    matches = movies[movies['title'].str.lower().str.contains(movie_name)]

    if matches.empty:
        return None

    movie_id = matches.iloc[0]['movieId']

    similar_movies = similarity_df[movie_id].sort_values(ascending=False)

    similar_movies = similar_movies.iloc[1:n_recommendations+1]

    recommended_ids = similar_movies.index.tolist()

    return movies[movies['movieId'].isin(recommended_ids)]['title'].tolist()

