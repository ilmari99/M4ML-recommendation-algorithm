import heapq
import json
import os
import pickle
import random
from typing import Callable
import numpy as np
from Movie import Movie
from User import User
from svd_approx import svd_approx_error
import matplotlib.pyplot as plt
from utils import read_movies, read_ratings, create_user_movie_matrix, get_R_and_TM, get_id_conversions, get_users_movies
from sklearn.decomposition import NMF

def svd_approx(U, S, V, k):
    """ Returns the SVD approximation of the matrix """
    U = U[:,0:k]
    S = np.diag(S[0:k])
    V = V[0:k,:]
    return U @ S @ V

def centered_cosine_similarity(u, k):
    """ Returns the centered cosine similarity between two vectors."""
    u = u - np.mean(u, axis=0, keepdims=True)
    k = k - np.mean(k, axis=0, keepdims=True)
    return np.dot(u, k) / (np.linalg.norm(u) * np.linalg.norm(k))

def find_k_closest(u, R, watched, axis = "auto", k = 10, similarity_metric = None) -> tuple:
    """ Returns the k closest (cosine) vectors to u in R """
    if axis == "auto":
        try:
            axis = np.where(R.shape == len(u))[0][0]
        except:
            raise Exception("Could not find axis to use")
    if similarity_metric is None:
        similarity_metric = centered_cosine_similarity
    if not isinstance(similarity_metric, Callable):
        raise ValueError("similarity_metric must be callable")
    # Compare the rows of R to u
    if axis == 1:
        # Find the cosine similarity between u and each column in R
        similarities = [(i,similarity_metric(u, R[:,i])) for i in watched]
    if axis == 0:
        # Find the cosine similarity between u and each row in R
        similarities = [(i,similarity_metric(u, R[i,:])) for i in watched]
    else:
        raise ValueError("Axis must be 0 or 1, but was " + str(axis))
    if not similarities:
        print(f"User has not rated any movies.")
        return [], []
    # Find the k largest similarities
    #k_iter = ((i, sim) for i, sim in enumerate(similarities))
    #k_largest = heapq.nlargest(k, similarities)
    k_sorted = sorted(similarities, key=lambda x: x[1], reverse=True)
    # Find the indices of the k largest similarities
    k_largest_indices = [i for i, sim in k_sorted[0:k]]
    k_largest = [sim for i, sim in k_sorted[0:k]]
    #k_largest_indices = np.argpartition(np.array(similarities), -k)[-k:]
    #k_largest = [similarities[i] for i in k_largest_indices]
    return k_largest_indices, k_largest

def get_U_M(R, k=10):
    """ Initialises two random matrices U (users x k) and M (k x movies).
    Finds a non-negative matrix factorisation of R = U @ M.
    """
    # Initialise U and M
    U = np.random.rand(R.shape[0], k)
    M = np.random.rand(k, R.shape[1])
    error = float("inf")
    # Perform gradient descent
    while error > 1020:
        # Calculate the errors matrix
        error = R - U @ M
        # Calculate the gradients
        U_grad = -error @ M.T
        M_grad = -U.T @ error
        # Apply the gradients
        U -= 0.0001 * U_grad
        M -= 0.0001 * M_grad
        error = np.linalg.norm(error)
        print(f"Error: {error}")
    print(f"U: {U.shape}, M: {M.shape}")
    print(f"U: {U},\n M: {M}")
    return U, M

def sklearn_NMF_U_M(R, k=10, max_iter=500, regularization=1):
    """ Uses the sklearn NMF algorithm to find a non-negative matrix factorisation of R = U @ M.
    """
    model = NMF(n_components=k, init='random', random_state=0,max_iter=max_iter,alpha_H=regularization,alpha_W=regularization, verbose=1)
    U = model.fit_transform(R)
    M = model.components_
    return U,M

# Do a dimensionality reduction of each movies rating vector to k dimensions with SVD
# For each test user, and for each ot their unrated movie:
#   Find the k most similarly rated movies and the rating by the user for each
#   Find the weighted (similarity) average of the users ratings for the k most similar movies
#   Fill in the rating for the un rated movie
#   Repeat for all unrated movies
if __name__ == "__main__":
    # define some constants
    read_nlines = "all"
    test_percent = 0.1
    kn = 2 #  how many neighbors to use for the weighted average
    k = 2 #  which rank to use for the SVD approximation
    random.seed(42)

    movieid_to_sequential_movieid, sequential_movieid_to_movieid, userid_to_sequential_userid, sequential_userid_to_userid = get_id_conversions()
    R, TM = get_R_and_TM(R_file = "R1.npy", TM_file = "TM1.npy",save=True, test_size=test_percent)
    movies, users = get_users_movies(movie_file="movies.pkl", rating_file="users.pkl", from_pickle=True)

    #R, TM = get_R_and_TM(R_file = "R1.npy", TM_file = "TM1.npy",save=False, test_size=test_percent)
    # Center the user ratings. Subtract each row by the mean of the GIVEN ratings (so 0 wont be accounted for)
    #R = R - np.where(R != 0, R.mean(axis=1, keepdims=True), 0)
    #TM = TM - np.where(TM != 0, TM.mean(axis=1, keepdims=True), 0)
    #R_means = R.mean(axis = 0, keepdims=True)
    #TM_means = TM.mean(axis = 0, keepdims=True)
    #R = R - R_means
    #TM = TM - TM_means
    #print(f"User means = {R_means}")
    # Approximate the matrix using SVD
    #U, S, V = np.linalg.svd(R)
    U, V = sklearn_NMF_U_M(R, k=k)
    #U,V = get_U_M(R, k=k)
    # V_k is (movies x k)
    V_k = V[0:k,:].T
    
    errs = []
    for i, new_user in enumerate(TM):
        new_user_preds = []
        # Store the original users ratings. Mark half of the actual ratings as 0, so we can test the predictions
        original_ratings = new_user.copy()
        new_user = np.array([rating if random.random() > 0.5 else 0 for rating in new_user])
        
        # Get the indices of the movies the user has rated
        watched = [i for i, rating in enumerate(new_user) if rating != 0]
        print(f"User {i} has rated {len(watched)} movies.")
        for movie in range(len(new_user)):
            # If the user has not rated the movie
            if new_user[movie] == 0:
                # Find the k most similar movies to the movie
                k_most_similar_movies, similarity = find_k_closest(V_k[movie,:], V_k, watched, axis=0, k=kn, similarity_metric=centered_cosine_similarity)
                # Find the rating for each of the k most similar movies
                k_most_similar_movies_ratings = [new_user[m] for m in k_most_similar_movies]
                # Find the weighted average of the ratings
                pred = np.dot(similarity, k_most_similar_movies_ratings) / sum([abs(s) for s in similarity])
                if pred > 5.001:
                    print("Predicted rating is greater than 5")
                    print(f"Movie rating: {new_user[movie]} predicted rating: {pred}")
                    print(f"Similarities: {similarity}")
                    print(f"Ratings: {k_most_similar_movies_ratings}")
                new_user_preds.append(pred)
            else:
                new_user_preds.append(new_user[movie])
            #print(f"Movie {movie} rating: {new_user[movie]} predicted rating: {new_user_preds[movie]}")
        movie_rating_pairs = [(m, r) for m, r in enumerate(new_user)]
        most_liked_movies = sorted(movie_rating_pairs, reverse=True,key=lambda x : x[1])[0:5]
        most_liked_movies_indices = [m[0] for m in most_liked_movies]
        print("Most liked movies:", [sequential_movieid_to_movieid[str(m)] for m in most_liked_movies_indices])
        print("Which are: ")
        for i, m in enumerate(most_liked_movies_indices):
            print(f"{i+1}) {movies[m]}, rating: {most_liked_movies[i][1]}")
        predicted_ratings = np.array(new_user_preds) - new_user
        predicted_rating_index_pairs = [(i, r) for i, r in enumerate(predicted_ratings)]
        most_predicted_ratings = sorted(predicted_rating_index_pairs, reverse=True,key=lambda x : x[1])[0:5]
        most_predicted_ratings_indices = [m[0] for m in most_predicted_ratings]
        print("Movies predicted to like most:", [sequential_movieid_to_movieid[str(m)] for m in most_predicted_ratings_indices])
        print("Which are: ")
        for i, m in enumerate(most_predicted_ratings_indices):
            print(f"{i+1}) {movies[m]}, rating: {most_predicted_ratings[i][1]}")
        # Compare the predicted ratings to the actual ratings
        # We can't say how good a prediction is, if the user has not rated the movie
        # So we will only compare the movies that the user has rated, but which were not used to make the prediction

        # Get the indices of the movies the user has rated
        watched = [i for i, rating in enumerate(original_ratings) if rating != 0]
        # Get the indices of the movies the user has rated, but which were not used to make the prediction
        watched_but_not_used = [i for i, rating in enumerate(original_ratings) if rating != 0 and new_user[i] == 0]
        # Get the predicted ratings for the movies the user has rated, but which were not used to make the prediction
        predicted_ratings = [new_user_preds[i] for i in watched_but_not_used]
        # Get the actual ratings for the movies the user has rated, but which were not used to make the prediction
        actual_ratings = [original_ratings[i] for i in watched_but_not_used]
        # Calculate the mean squared error
        errs += (np.array(predicted_ratings) - np.array(actual_ratings)).tolist()
        mse = np.mean((np.array(predicted_ratings) - np.array(actual_ratings))**2)
        print(f"Watched but not used ratings: {actual_ratings}")
        print(f"Predicted ratings: {[round(r,2) for r in predicted_ratings]}")
        print(f"Mean squared error of user: {mse}")
        print(f"Mean squared error of all users: {np.mean(np.power(errs,2))}")
        print("")




