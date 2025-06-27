import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate
from collections import defaultdict
import numpy as np

print("Building a Recommendation System using Matrix Factorization (SVD).")
print("This system will recommend movies based on user ratings.")

# --- 1. Load Data ---
# We will use the MovieLens 100k dataset, which is conveniently available in Surprise.
# It contains 100,000 ratings (1-5) from 943 users on 1682 movies.

# The Reader object is used to parse a file containing ratings.
# The 'line_format' parameter indicates that each line contains
# user, item, and rating, separated by a tab ('\t').
# The 'rating_scale' parameter indicates that ratings are between 1 and 5.
print("\nLoading MovieLens 100k dataset...")
reader = Reader(line_format='user item rating timestamp', sep='\t')

# Load the dataset from the built-in MovieLens 100k files.
# Dataset.load_from_folder downloads the data if not present.
# Note: MovieLens 100k dataset structure has 4 columns: userId, movieId, rating, timestamp.
data = Dataset.load_from_builtin('ml-100k')

print("Dataset loaded successfully.")

# --- 2. Split Data for Training and Testing ---
# We'll split the data into a training set and a testing set.
# The `surprise` library handles this internally for cross-validation,
# but it's good to see how to manually split for a simple train/test evaluation.
print("\nSplitting data into training and testing sets...")
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)
print(f"Training set size: {len(trainset.ur)} ratings") # Number of user ratings in trainset
print(f"Test set size: {len(testset)} ratings")

# --- 3. Choose a Model: Singular Value Decomposition (SVD) ---
# SVD is a matrix factorization technique that decomposes the user-item rating matrix
# into a product of two lower-dimensional matrices: a user-feature matrix and an item-feature matrix.
# These 'features' are latent factors that represent underlying preferences.
print("\nInitializing the SVD model...")
algo = SVD(random_state=42)

# --- 4. Train the Model ---
print("Training the SVD model...")
algo.fit(trainset)
print("Model training complete.")

# --- 5. Evaluate the Model (on test set) ---
# Predict ratings for the test set and calculate evaluation metrics.
print("\nEvaluating the model on the test set...")
predictions = algo.test(testset)

# We can evaluate using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
# RMSE gives a relatively high weight to large errors.
# MAE is less sensitive to outliers.
from surprise import accuracy
print("Calculating evaluation metrics:")
accuracy.rmse(predictions, verbose=True) # Root Mean Squared Error
accuracy.mae(predictions, verbose=True)  # Mean Absolute Error

# --- 6. Cross-Validation (More Robust Evaluation) ---
# Cross-validation provides a more robust estimate of model performance
# by training and testing on different folds of the data.
print("\nPerforming 5-fold cross-validation...")
cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print("\nCross-validation results:")
print(f"Average RMSE: {np.mean(cv_results['test_rmse']):.4f}")
print(f"Average MAE: {np.mean(cv_results['test_mae']):.4f}")
print("Cross-validation complete.")

# --- 7. Make a Prediction for a Specific User and Item ---
print("\n--- Making Individual Predictions ---")
# Let's pick a user and an item that are likely in our dataset.
# User ID '196' and Item ID '242' are common in MovieLens 100k.
user_id = '196'
item_id = '242' # This corresponds to the movie "Kolya (1996)"

# The `predict` method estimates the rating that user_id would give to item_id.
# The 'r_ui' parameter is the true rating if known (for evaluation), otherwise set to None.
# Here, we don't know the true rating for a new prediction.
# If we knew it, it would be e.g., r_ui=4.0
predicted_rating = algo.predict(user_id, item_id)
print(f"Predicted rating for user '{user_id}' on item '{item_id}': {predicted_rating.est:.2f}")

# Example with a known rating from the test set for comparison:
# Find a true rating from the test set to compare with a prediction
# (This is illustrative; in a real scenario, you'd predict for items not yet rated)
if testset:
    # Get a sample from the testset
    sample_test_rating = testset[0]
    sample_user, sample_item, sample_true_rating = sample_test_rating
    predicted_for_sample = algo.predict(sample_user, sample_item)
    print(f"For user '{sample_user}' on item '{sample_item}':")
    print(f"  True rating: {sample_true_rating:.2f}")
    print(f"  Predicted rating: {predicted_for_sample.est:.2f}")


# --- 8. Generate Top-N Recommendations for a User ---
# First, get a list of all movies that the target user has NOT yet rated.
# Then, predict ratings for these unrated movies and recommend the top N.

def get_top_n_recommendations(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions."""

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

print(f"\n--- Generating Top-N Recommendations for User '{user_id}' ---")

# To get unrated items for a specific user, we need the full dataset.
# We'll re-train the model on the full dataset for making real-world recommendations
# (as opposed to evaluation on a held-out test set).
full_trainset = data.build_full_trainset()
full_algo = SVD(random_state=42)
full_algo.fit(full_trainset)

# Get a list of all item IDs
all_item_ids = full_trainset.all_items()
# Map raw item IDs to inner item IDs (used by Surprise internally)
item_id_to_name_map = full_trainset.raw_item_ids

# Get a list of items already rated by the target user
user_rated_items = set([iid for (iid, _) in full_trainset.ur[full_trainset.to_inner_uid(user_id)]])

# Create a list of items not yet rated by the user
unrated_items = []
for item_inner_id in all_item_ids:
    if item_inner_id not in user_rated_items:
        unrated_items.append(full_trainset.to_raw_iid(item_inner_id))


# Predict ratings for all unrated items for the target user
user_predictions = [full_algo.predict(user_id, item_id) for item_id in unrated_items]

# Sort the predictions by estimated rating in descending order
user_predictions.sort(key=lambda x: x.est, reverse=True)

# Print the top 10 recommendations
n_recommendations = 10
print(f"Top {n_recommendations} recommendations for user '{user_id}':")

# To make this more user-friendly, let's load movie titles.
# In a real application, you'd have a database or file mapping item IDs to names.
# For MovieLens 100k, we can parse the u.item file.
try:
    movies_df = pd.read_csv(
        Dataset.load_builtin('ml-100k').path + 'u.item',
        sep='|',
        names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
               'unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy',
               'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror',
               'musical', 'mystery', 'romance', 'scifi', 'thriller', 'war', 'western'],
        encoding='latin-1'
    )
    # Create a mapping from movie_id (string) to title
    movie_titles = {str(row['movie_id']): row['title'] for index, row in movies_df.iterrows()}

    for i, pred in enumerate(user_predictions[:n_recommendations]):
        movie_title = movie_titles.get(pred.iid, f"Item ID {pred.iid} (Title Not Found)")
        print(f"  {i+1}. {movie_title} (Estimated rating: {pred.est:.2f})")
except Exception as e:
    print(f"Could not load movie titles for display: {e}")
    print("Displaying recommendations by Item ID only:")
    for i, pred in enumerate(user_predictions[:n_recommendations]):
        print(f"  {i+1}. Item ID: {pred.iid} (Estimated rating: {pred.est:.2f})")


print("\nRecommendation system implementation complete!")
