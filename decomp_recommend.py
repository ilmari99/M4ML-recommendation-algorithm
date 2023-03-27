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

def svd_U_M(R, k=10):
    """ Returns the matrices U and M, where R = U @ M. """
    U, S, V = np.linalg.svd(R)
    V = np.diag(S[0:k]) @ V[0:k,:]
    return U[:,0:k], V

def als_U_M(R, k=10, max_iters=1000, patience=10, threshold=0.1, plot=False):
    """ Initialises two matrices U (users x k) and M (k x movies) randomly.
    Finds a non-negative matrix factorisation of R = U @ M using alternating least squares.
    """
    # Initialise U and M
    U = np.random.rand(R.shape[0], k)
    M = np.random.rand(k, R.shape[1])
    all_errors = []
    errors = []
    pat = 0
    min_error = float("inf")
    # Perform alternating least squares
    for i in range(max_iters):
        if pat > patience:
            if min(errors) >= min_error - threshold:
                break
            else:
                min_error = min(errors)
                pat = 0
                errors = []
        # Update U
        M_t_M = M @ M.T
        #print(f"M_t_M: {M_t_M.shape}")
        for u in range(R.shape[0]):
            yy = [R[u, m] * M[:,m] for m in range(R.shape[1])]
            yy = np.sum(yy, axis=0)
            U[u] = np.linalg.lstsq(M_t_M, yy, rcond=None)[0]
        # Update M
        U_t_U = U.T @ U
        #print(f"U_t_U: {U_t_U.shape}")
        for m in range(R.shape[1]):
            yy = [R[u, m] * U[u,:] for u in range(R.shape[0])]
            yy = np.sum(yy, axis=0)
            M[:, m] = np.linalg.lstsq(U_t_U, yy, rcond=None)[0]
        # Make sure there are no negative values
        U[U < 0] = 0
        M[M < 0] = 0
        # Calculate the error
        error = np.linalg.norm(R - U @ M, ord="fro")
        print(f"Step {i}: Error: {error}",)
        errors.append(error)
        all_errors.append(error)
        pat += 1
    print(f"U: {U.shape}, M: {M.shape}")
    # Check if there are any negative values
    if np.any(U < 0) or np.any(M < 0):
        print("Negative values in U or M")
    if plot:
        plt.plot(all_errors)
        plt.title("Convergence of alternating least squares")
        plt.xlabel("Iteration")
        plt.ylabel("Reconstruction error (frobenius norm)")
        plt.show()
    return U, M

def grad_desc_U_M(R, k=10, lr=0.00001, threshold = 0.1, patience = 10, max_iters=1000, plot=False):
    """ Initialises two random matrices U (users x k) and M (k x movies).
    Finds a non-negative matrix factorisation of R = U @ M.
    """
    # Initialise U and M
    U = np.random.rand(R.shape[0], k)
    M = np.random.rand(k, R.shape[1])
    error = float("inf")
    # Perform gradient descent
    iters = 0
    pat = 0
    all_errors = []
    errors = []
    min_error = float("inf")
    while iters < max_iters:
        # Check if the error hasn't decreased sufficiently
        if pat >= patience:
            if min(errors) > min_error - threshold:
                break
            else:
                min_error = min(errors)
                pat = 0
                errors = []
        iters += 1
        # Calculate the errors matrix
        error = R - U @ M
        # Calculate the gradients
        U_grad = -2*error @ M.T
        M_grad = -2*U.T @ error
        # Apply the gradients
        U -= lr * U_grad
        M -= lr * M_grad
        # Make sure there are no negative values
        U[U < 0] = 0
        M[M < 0] = 0
        error = np.linalg.norm(error, ord="fro")
        errors.append(error)
        all_errors.append(error)
        pat += 1
        print(f"Error: {error}")
    print(f"U: {U.shape}, M: {M.shape}")
    # Check if there are any negative values
    if np.any(U < 0) or np.any(M < 0):
        print("Negative values in U or M")
    if plot:
        plt.plot(all_errors)
        plt.title("Convergence of gradient descent")
        plt.xlabel("Iteration")
        plt.ylabel("Reconstruction error (frobenius norm)")
        plt.show()  
    return U, M

def remove_ratings(R, split=0.1):
    """ Removes a percentage of the ratings from the matrix R.
    Returns a matrix R_test with only the removed ratings and a matrix R, where the corresponding values have been set to 0.
    """
    # Get the number of ratings to remove
    n_ratings = np.count_nonzero(R)
    print(f"Number of ratings: {n_ratings}")
    n_ratings_to_remove = int(n_ratings * split)
    print(f"Removing {n_ratings_to_remove} ratings...")
    # Get the indices of the non-zero elements
    rs, cols = np.nonzero(R)
    # Take a random sample of the indices
    ratings_to_remove = random.sample(list(zip(rs,cols)),k=n_ratings_to_remove)
    # Create a new matrix to store the removed ratings
    removed_ratings = np.zeros_like(R)
    done_ratings = set()
    # Remove the ratings and store them in removed_ratings
    for i,j in ratings_to_remove:
        # Check if we have already removed this rating
        if (i,j) in done_ratings:
            print(f"Already removed rating at ({i},{j})")
            continue
        done_ratings.add((i,j))
        removed_ratings[i,j] = R[i,j]
        R[i,j] = 0
    print(f"Number of ratings removed: {np.count_nonzero(removed_ratings)}")
    return R, removed_ratings

def fill_estimates(R):
    """
    # Fill in the missing values in R with a global estimate
    # The estimate consists of:
    # 1. The mean of all ratings (global mean)
    # 2. The mean of the ratings for the user (user mean)
    # 3. The mean of the ratings for the movie (movie mean)
    # The estimate, that a user i gives to a movie j,
    # is the global mean + the difference between the user mean and the global mean + the difference between the movie mean and the global mean
    """
    R_mean = np.ma.masked_equal(R, 0).mean(axis=1)
    C_mean = np.ma.masked_equal(R, 0).mean(axis=0)
    mean_all = np.ma.masked_equal(R, 0).mean()
    R = np.where(R == 0, mean_all + (C_mean[np.newaxis, :] - mean_all) + (R_mean[:, np.newaxis] - mean_all), R)
    return R


def get_predicted_values(R_approx, removed_ratings):
    """ Returns the values in R_approx, that were removed from R.
    """
    predicted_values = np.zeros_like(R)
    r_rows, r_cols = np.nonzero(removed_ratings)
    for inds in zip(r_rows, r_cols):
        i,j = inds
        predicted_values[i,j] = R_approx[i,j]
    return predicted_values

def main(R, k, method = "svd", split=0.1,method_kwargs={}, plot=False):
    """ The main function. Runs the desired method for matrix factorization.
    It prints the Frobenius norm of the approximate matrix, and the MSE and MAE of the predictions.

    Parameters
    ----------
    R : np.array
        The matrix to be factorized.
    k : int | list[int]
        The rank of the matrix factorization. If a list is given, the method is run for each value in the list.
    method : str, optional
        The method to be used for matrix factorization. The default is "svd".
    split : float, optional
        The percentage of ratings to be removed from the matrix. The default is 0.1.
    method_kwargs : dict, optional
        Additional keyword arguments for the method. The default is {}.
    plot : bool, optional
        Whether to plot the errors. The default is False. This only works, if k is a list.
    """
    if plot and not isinstance(k, list):
        raise ValueError("k must be a list, if plot is True")
    if not isinstance(k, list):
        k = [k]
    R_og = R.copy()
    errors = []
    for k_ in k:
        R = R_og.copy()
        R, removed_ratings = remove_ratings(R, split=split)
        R = fill_estimates(R)
        if method == "svd":
            U, M = svd_U_M(R, k=k_, **method_kwargs)
        elif method == "als":
            U, M = als_U_M(R, k=k_, **method_kwargs)
        elif method == "gd":
            U, M = grad_desc_U_M(R, k=k_, **method_kwargs)
        else:
            raise ValueError(f"Method {method} not implemented")
        
        # Check if U and M are non-negative
        if np.any(U < 0) or np.any(M < 0):
            print("U or M are negative")
        R_approx = U @ M
        err = np.linalg.norm(R_approx - R, ord="fro")
        print(f"Reconstruction error of R: {err}")
        predicted_values = get_predicted_values(R_approx, removed_ratings)
        nz_r, nz_c = np.nonzero(removed_ratings)
        nz_inds = list(zip(nz_r, nz_c))
        mse = np.mean([(removed_ratings[i,j] - predicted_values[i,j])**2 for i,j in nz_inds])
        mae = np.mean([np.abs(removed_ratings[i,j] - predicted_values[i,j]) for i,j in nz_inds])
        print(f"MSE (k={k_}): {mse}")
        print(f"MAE (k={k_}): {mae}")
        errors.append((err,mse, mae))
    if plot:
        methods = {"svd": "SVD", "als": "ALS", "gd": "Gradient descent"}
        errors = np.array(errors)
        for i in range(3):
            fig, ax = plt.subplots()
            ax.plot(k, errors[:,i])
            ax.set_xlabel("k")
            ax.set_ylabel(["Frob norm", "Test MSE", "Test MAE"][i])
            ax.set_title([f"Reconstruction error of R", f"MSE of the removed values", f"MAE of the removed values"][i] + f" using {methods[method]}")
        plt.show()







# Shape of R: (nusers, nmovies)
# Shape of U: (nusers, k)
# Shape of M: (k, nmovies)

# In this approach, we do not calculate similarities or neigbours (knn)
# Create a matrix R representing the ratings of users for movies
# R is a matrix of shape (nusers, nmovies)
# R[i,j] is the rating of user i for movie j
# R[i,j] = 0 if user i has not rated movie j

# We want to find a matrix factorisation of R = U @ M
# k is the rank of the matrix factorisation
# U is a matrix of shape (nusers, k)
# M is a matrix of shape (k, nmovies)
# U[i,:] is the vector representation of user i
# M[:,j] is the vector representation of movie j
# U[i,:] @ M[:,j] is the predicted rating of user i for movie j

# We want to find U and M such that R is approximated as closely as possible
# We can do this by minimising the error between R and U @ M

# The approach:
# Create matrix R and remove 10 % of the ratings (and store them), call this matrix R_sub
# Find the matrix factorisation, where min ||R_sub - U @ M||^2
# Reconstruct matrix R_sub by multiplying U and M, call this matrix R_pred
# Measure the error between R_sub and R_pred
# See what are the predicted values for the removed ratings in R_pred
# Measure the MSE and MAE of the predicted values for the removed ratings
# Repeat for different values of k and different algorithms

if __name__ == "__main__":
    # define some constants
    test_percent = 0.1
    k = [i for i in range(1,11)] #  which rank to use for the SVD approximation
    k = 4
    random.seed(42)
    R = np.load("R.npy")
    # gd args: "lr":0.00001, "threshold":1, "patience": 10, "max_iters":1000
    # als args: "threshold":1, "patience": 2, "max_iters":30
    # Run the main function
    #0.685
    main(R, k, method="als", split=test_percent, method_kwargs={"threshold":1, "patience": 2, "max_iters":30}, plot=False)
    
    



    





