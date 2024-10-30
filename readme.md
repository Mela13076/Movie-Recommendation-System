
# Movie Recommendation System

This project implements a content-based movie recommendation system using data from **The Movies Dataset** on Kaggle. The system suggests movies similar to a given input movie based on content attributes like genres, keywords, and plot overviews.

## Dataset

The dataset used in this project is [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) from Kaggle. It contains extensive metadata on movies, including information on genres, keywords, and movie descriptions, which is ideal for content-based filtering.

### Files Used
- `movies_metadata.csv`: Contains metadata about movies such as title, genres, overview, and release dates.
- `keywords.csv`: Contains keywords associated with each movie.

## Project Overview

The recommendation system works as follows:
1. **Data Loading and Preprocessing**: 
   - Load `movies_metadata.csv` and `keywords.csv`, convert relevant fields into usable formats, and merge them.
   - Combine genres, keywords, and overview text for each movie into a single text feature for content analysis.
2. **TF-IDF Vectorization**:
   - Use TF-IDF (Term Frequency-Inverse Document Frequency) to represent the combined movie features numerically, capturing the importance of each term within the dataset.
3. **Cosine Similarity Calculation**:
   - Calculate the cosine similarity between movies to measure how closely related they are based on the TF-IDF features.
4. **Recommendation Function**:
   - Given a movie title, the function finds the top 5 most similar movies based on genre and similarity score.
5. **Visualization**:
   - Display the similarity scores of the recommended movies using a bar chart.

## How to Use the Code

### Prerequisites
1. **Python 3.9+**
2. **Libraries**:
   - Install required libraries with:
     ```bash
     pip install pandas scikit-learn numpy matplotlib seaborn
     ```

### Steps to Run
1. **Download the Dataset**:
   - Download [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) from Kaggle and place `movies_metadata.csv` and `keywords.csv` in the `datasets/` directory within your project.

2. **Run the Script**:
   - To run the code, execute:
     ```bash
     python main.py
     ```
   - Replace `"Toy Story"` in the code with the title of any movie you want recommendations for.

3. **Expected Output**:
   - The script will output the top 5 recommended movies based on the input title.
   - A bar chart will display the cosine similarity scores of the recommendations.

### Example Usage

To get recommendations for *Toy Story*, use:
```python
movie = "Toy Story"
recommended_titles, similarity_scores = get_recommendations(movie)
```

Expected output:
- Recommended titles (e.g., *Toy Story That Time Forgot*, *Small Fry*, etc.)
- A bar chart showing similarity scores.

## Explanation of Key Components

- **Content-Based Filtering**: This approach recommends movies based on metadata such as genres, keywords, and plot overviews. By analyzing these features, the system identifies movies with similar content.
- **TF-IDF Vectorization**: A technique to quantify text data by assigning weights based on term frequency and rarity across documents, highlighting important words.
- **Cosine Similarity**: Measures the similarity between two movies based on their TF-IDF vectors, resulting in scores that represent how closely related each movie is to the input.

## Acknowledgements

Special thanks to [Kaggle](https://www.kaggle.com) for providing [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset), which served as the foundation for this recommendation system.

