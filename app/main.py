from fastapi import FastAPI
from app.recommender import recommend_movies

app = FastAPI(title="Movie Recommendation API")

@app.get("/")
def home():
    return {"message": "Movie Recommender API is running 🚀"}

@app.get("/recommend")
def recommend(movie: str):
    results = recommend_movies(movie)
    if results is None:
        return {"error": "Movie not found"}
    return {
        "movie": movie,
        "recommendations": results
    }
@app.get("/movies")
def list_movies():
    return movies['title'].sample(20).tolist()



