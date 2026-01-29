from fastapi import FastAPI
from app.recommender import recommend_movies

app = FastAPI(title="Movie Recommendation API")

@app.get("/")
def home():
    return {"message": "Movie Recommender API is running 🚀"}

@app.get("/recommend")
def recommend(movie: str, top_n: int = 5):
    recommendations = recommend_movies(movie, top_n)

    if not recommendations:
        return {"error": "Movie not found"}

    return {
        "input_movie": movie,
        "recommendations": recommendations
    }
