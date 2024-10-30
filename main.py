# Step 1: Import Libraries and Load Data
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the main movies metadata file and keywords file
movies = pd.read_csv('./datasets/movies_metadata.csv', low_memory=False)
# Sample a subset of movies (for example, 5,000 movies out of 45,000)
movies = movies.sample(12000, random_state=42)
keywords = pd.read_csv('datasets/keywords.csv')

# Convert both 'id' columns to strings to ensure consistent data types for merging
movies['id'] = movies['id'].astype(str)
keywords['id'] = keywords['id'].astype(str)

# Merge keywords into the main movie dataset on the 'id' field
movies = movies.merge(keywords[['id', 'keywords']], on='id', how='left')

# Step 2: Data Preprocessing

# Convert genres and keywords from JSON format to lists of strings
def convert_json_to_list(text):
    try:
        # Parse the JSON-like string to a list of dictionaries
        items = ast.literal_eval(text)
        # Extract the name for each genre/keyword and return as a list
        return [item['name'] for item in items]
    except (ValueError, SyntaxError):
        return []

# Apply the function to genres and keywords columns
movies['genres'] = movies['genres'].fillna('[]').apply(convert_json_to_list)
movies['keywords'] = movies['keywords'].fillna('[]').apply(convert_json_to_list)

# Fill NaN values in 'overview' and combine all features into a single string
movies['overview'] = movies['overview'].fillna('')
movies['combined_features'] = movies['genres'].apply(lambda x: ' '.join(x)*2) + ' ' + \
                              movies['keywords'].apply(lambda x: ' '.join(x)*2) + ' ' + \
                              movies['overview']

# Step 3: TF-IDF Vectorization
# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)

# Create the TF-IDF matrix for the combined features
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# Step 4: Calculate Cosine Similarity
# Compute cosine similarity matrix from the TF-IDF matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim = np.nan_to_num(cosine_sim)

# Step 5: Create a Movie Recommendation Function

# Create a reverse mapping of movie titles to indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

#Step 6: Test the Recommendation System
def get_recommendations(title, cosine_sim=cosine_sim):

    if title not in indices:
        print(f"Movie '{title}' not found in the dataset.")
        return None, None
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))

    valid_scores = []
    for score in sim_scores:
        value = score[1][0] if isinstance(score[1], (np.ndarray, list)) else score[1]
        if not np.isnan(value) and value > 0.1:
            valid_scores.append((score[0], value))

    valid_scores = sorted(valid_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [
        i[0] for i in valid_scores[1:]
        if any(genre in movies.iloc[i[0]]['genres'] for genre in movies.loc[idx, 'genres'])
    ][:5]
    scores = [score[1] for score in valid_scores[1:6]]  # Top 5 similarity scores

    # Return titles and similarity scores
    return movies['title'].iloc[movie_indices].tolist(), scores

movie = "Toy Story"
#suggested movies to try: Jumanji, Toy Story,  Mortal Kombat
recommended_titles, similarity_scores = get_recommendations(movie)

if not recommended_titles == None or  not similarity_scores == None: 
    print(recommended_titles)

    # Bar plot for similarity scores
    plt.figure(figsize=(10, 6))
    plt.barh(recommended_titles[::-1], similarity_scores[::-1], color='skyblue')
    plt.xlabel("Cosine Similarity Score")
    plt.title(f"Similarity of Recommended Movies to {movie}")
    plt.gca().invert_yaxis()  # Invert y-axis for highest score at the top
    plt.show()