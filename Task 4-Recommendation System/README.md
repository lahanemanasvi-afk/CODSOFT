# Project Title: Content-Based Movie Recommendation System

## CodSoft Internship Task 4: Recommendation System

### I. Introduction and Objective

This project, developed as part of the CodSoft Internship, implements a foundational **Content-Based Filtering** recommendation system. The core objective is to suggest movies to a user based purely on the descriptive characteristics (content) of a movie they have already shown interest in.

Unlike collaborative filtering which uses user ratings, this system relies on calculating the inherent similarity between the content attributes of two different movies.

### II. Working Methodology

The Content-Based Filtering process is executed through a sequence of well-defined data science steps:

#### A. Data Definition and Preprocessing

* **Dataset:** A small, custom dataset is defined using a Pandas DataFrame. Each entry contains a `title` and a collection of descriptive `tags` (genres, keywords, themes).
    * *Example Tags:* 'Sci-Fi Space Exploration Future Survival Time Travel', 'Action Crime Thriller Superhero'.
* **Data Structure:** The data is structured to be compatible with vectorization techniques.

#### B. Feature Engineering with TF-IDF Vectorizer

* **Technique:** **TF-IDF (Term Frequency-Inverse Document Frequency)** is used to convert the raw text tags into a numerical feature matrix.
* **Purpose:** TF-IDF assigns a weighted value to each word (token) in the tags. A word that appears frequently in one movie's tags but rarely across all movies is given a higher weight, indicating its significance in defining that specific movie's content.
* **Output:** The `TfidfVectorizer` creates a sparse matrix where rows represent movies and columns represent unique keywords/tokens, with values indicating the TF-IDF weight.

#### C. Similarity Calculation (Cosine Similarity)

* **Metric:** **Cosine Similarity** is employed to determine the degree of similarity between any two movies in the feature matrix.
* **Mechanism:** It measures the cosine of the angle between two multi-dimensional vectors.
    * A cosine value of **1.0** indicates that the two movie vectors are pointing in exactly the same direction (identical content).
    * A value of **0.0** indicates that the vectors are orthogonal (no shared content).
* **Output:** A symmetrical matrix (e.g., 10x10 for 10 movies) is generated, where the value at position `[i, j]` is the similarity score between movie `i` and movie `j`.

#### D. Recommendation Generation

* **Lookup:** The index of the input movie is retrieved.
* **Sorting:** The row corresponding to the input movie's index from the Cosine Similarity matrix is extracted and sorted in descending order of similarity score.
* **Filtering:** The input movie itself (which will have a similarity score of 1.0) is excluded.
* **Final Output:** The top N indices are converted back into movie titles, providing the final list of content-similar recommendations.

### III. Implementation Details (Technical Stack)

| Component | Technology / Library | Role in the Project |
| :--- | :--- | :--- |
| **Language** | Python | Core programming language. |
| **Data Handling** | `pandas` | Used for creating the initial DataFrame and handling data structures. |
| **Vectorization** | `sklearn.feature_extraction.text.TfidfVectorizer` | Converts text tags into numerical feature vectors. |
| **Core Algorithm** | `sklearn.metrics.pairwise.cosine_similarity` | Calculates the similarity scores between all movie pairs. |
| **Utility** | `numpy` | Supports the underlying numerical operations within scikit-learn. |

### IV. Execution and Results

The script is executed directly via the Python interpreter.