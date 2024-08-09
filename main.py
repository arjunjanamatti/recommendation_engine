import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
from surprise.model_selection import cross_validate

# Load the MovieLens dataset
# You can use your own dataset in CSV format, but here we'll use the built-in MovieLens dataset
data = Dataset.load_builtin('ml-100k')

# Use 75% of the data for training and the rest for testing
trainset, testset = train_test_split(data, test_size=0.25)

# Use SVD (Singular Value Decomposition) for matrix factorization
algo = SVD()

# Train the model on the training data
algo.fit(trainset)

# Evaluate the performance on the test data
predictions = algo.test(testset)
print("RMSE: ", rmse(predictions))

# Cross-validate the algorithm using cross-validation
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Making a prediction for a specific user and item
user_id = str(196)  # User ID, must be a string
item_id = str(302)  # Item ID, must be a string
pred = algo.predict(user_id, item_id, verbose=True)

# Get the top 10 recommendations for a specific user
def get_top_n_recommendations(predictions, n=10):
    # First map the predictions to each user.
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the n highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

top_n = get_top_n_recommendations(predictions, n=10)

# Print the recommended items for a specific user
user_id = str(196)
print(f"Top 10 recommendations for user {user_id}:")
for item_id, rating in top_n[user_id]:
    print(f"Item {item_id} with predicted rating {rating:.2f}")

