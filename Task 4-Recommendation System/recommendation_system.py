import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. DATA DEFINITION (Custom, Small Dataset) ---
# A custom dataset of movies with their 'tags' (genres + keywords)
# These tags are the 'content' based on which recommendations will be made.
movie_data = {
    'title': [
        'Interstellar', 'The Martian', 'Arrival', 'The Dark Knight', 
        'Inception', 'Pulp Fiction', 'La La Land', 'The Grand Budapest Hotel', 
        'The Avengers', 'Avatar'
    ],
    'tags': [
        'Sci-Fi Space Exploration Future Survival Time Travel',
        'Sci-Fi Space Exploration Mars Survival NASA',
        'Sci-Fi Alien Contact Linguistics Mysterious',
        'Action Crime Thriller Superhero Gotham Dark',
        'Sci-Fi Thriller Dreams Subconscious Heist',
        'Crime Drama Dark Comedy Non-Linear Story',
        'Musical Drama Romance Hollywood Jazz',
        'Comedy Adventure Drama Whimsical Europe',
        'Action Superhero Team Alien Invasion Marvel',
        'Sci-Fi Fantasy Alien World Adventure Visuals'
    ]
}

# Convert the dictionary into a Pandas DataFrame for easier handling
df = pd.DataFrame(movie_data)

# --- 2. FEATURE EXTRACTION (TF-IDF Vectorization) ---
# TfidfVectorizer converts the text tags into numerical vectors.
# Tfidf: Term Frequency-Inverse Document Frequency. It weighs words by importance.
tfidf = TfidfVectorizer(stop_words='english')

# Apply the vectorizer to the 'tags' column to create the feature matrix
tfidf_matrix = tfidf.fit_transform(df['tags'])

# The tfidf_matrix now holds the numerical representation of movie content.
# print(tfidf_matrix.shape) # e.g., (10, 35) means 10 movies and 35 unique keywords

# --- 3. CALCULATE SIMILARITY (Cosine Similarity) ---
# Cosine Similarity measures the cosine of the angle between two vectors.
# A score closer to 1 means higher similarity (the content is very alike).
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# print(cosine_sim.shape) # Output is a 10x10 matrix where each element [i, j] is the similarity score between movie i and movie j

# --- Helper: Map Movie Titles to Indices ---
# This step creates a reverse map of titles to indices for quick lookups
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# --- 4. RECOMMENDATION FUNCTION ---

def get_recommendations(title, cosine_sim=cosine_sim, df=df, indices=indices, top_n=5):
    """
    Generates a list of top_n recommended movies based on the input movie title.
    
    Args:
        title (str): The title of the movie the user likes.
        top_n (int): The number of recommendations to return.
    """
    if title not in indices:
        print(f"Error: Movie '{title}' not found in the dataset.")
        return []

    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the similarity scores for that movie with all other movies
    # Enumerate helps us keep track of the original movie index
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity score (highest score first)
    # The key=lambda x: x[1] sorts by the score (the second element in the tuple)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar movies (excluding the movie itself, hence [1:top_n+1])
    sim_scores = sim_scores[1:top_n+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Get the titles of the recommended movies
    recommendations = df['title'].iloc[movie_indices]
    
    # Get the similarity scores (optional, for debugging/display)
    scores = [f"{i[1]:.3f}" for i in sim_scores]

    print(f"\n--- Recommendations for: {title} ---")
    print(f"Based on tags: {df['tags'].iloc[idx]}\n")
    
    for rank, (rec_title, score) in enumerate(zip(recommendations, scores)):
        print(f"{rank + 1}. {rec_title} (Similarity Score: {score})")
    
    print("-" * 40)
    return list(recommendations)

# --- EXECUTION / TESTING ---

if __name__ == "__main__":
    
    print("🎬 Simple Content-Based Movie Recommender (CodSoft Task 4) 🎬")
    print("=" * 60)
    print("Available Movies: ", ", ".join(df['title'].tolist()))
    print("=" * 60)

    # Example 1: Suggesting movies similar to 'Interstellar' (Sci-Fi, Space)
    get_recommendations('Interstellar', top_n=5)
    
    # Example 2: Suggesting movies similar to 'The Dark Knight' (Action, Crime, Thriller)
    get_recommendations('The Dark Knight', top_n=3)
    
    # Example 3: Suggesting movies similar to 'La La Land' (Musical, Romance, Jazz)
    get_recommendations('La La Land', top_n=4)
    
    # Example 4: Suggesting movies similar to 'Inception' (Sci-Fi, Dreams, Heist)
    get_recommendations('Inception', top_n=5)