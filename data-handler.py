import random
import numpy as np
from Movie import Movie
from User import User
from svd_approx import svd_approx_error
import matplotlib.pyplot as plt

def read_movies(filename, nlines="all") -> list[Movie]:
    """ Reads the movies from the file and returns a list of Movie objects

    movieId,title,genres
    1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
    2,Jumanji (1995),Adventure|Children|Fantasy
    3,Grumpier Old Men (1995),Comedy|Romance
    4,"The good, the bad, and the ugly",Comedy|Drama|Romance
    """
    movies = []
    ID = None
    title = None
    genres = []
    if nlines == "all":
        nlines = float("inf")
    with open(filename, "r",encoding="utf-8") as f:
        for line in f:
            # Skip the header
            if line.startswith("movieId"):
                continue
            if len(movies) >= nlines:
                break
            sp = line.strip().split(",")
            if len(sp) == 3:
                ID, title, genres = sp
            elif len(sp) > 3:
                ID = sp[0]
                genres = sp[-1]
                title = get_title_from_movie_data(line)
            ID = int(ID)
            genres = genres.split("|")
            movie = Movie(ID, title, genres)
            movies.append(movie)
            print(movie)
    return movies


def get_title_from_movie_data(line : str) -> str:
    """ Returns the title from the movie data line """
    first_quote = line.find('"')
    last_quote = line.rfind('"')
    if first_quote == -1 or last_quote == -1:
        return line.split(",")[1]
    return line[first_quote+1:last_quote]


def read_ratings(filename, nlines = "all") -> list[User]:
    """ Reads the ratings from the file and returns a list of User objects
    
    userId,movieId,rating,timestamp
    1,1,4.0,964982703
    1,3,4.0,964981247
    1,6,4.0,964982224
    1,47,5.0,964983815
    """
    users = {}
    read_lines = 0
    if nlines == "all":
        nlines = float("inf")
    with open(filename, "r") as f:
        for line in f:
            # Skip the header
            if line.startswith("userId"):
                continue
            if read_lines >= nlines:
                break
            read_lines += 1
            # Split the line into the userId, movieId, rating, and timestamp
            userId, movieId, rating, timestamp = line.strip().split(",")
            userId = int(userId)
            movieId = int(movieId)
            rating = float(rating)
            # Create a User object if it doesn't exist
            if userId not in users.keys():
                print(f"Creating user {userId}")
                users[userId] = User(userId)
            # Add the movie rating to the user
            users[userId].add_movie_rating(movieId, rating)
    return list(users.values())

def create_user_movie_matrix(users : list[User], movies : list[Movie]) -> np.ndarray:
    """ Creates a matrix of user ratings for each movie

    The rows are the users and the columns are the movies
    """
    # Create a matrix of zeros
    A = np.zeros((len(users), len(movies)))
    # Fill in the matrix with the user ratings
    for i, user in enumerate(users):
        for j, movie in enumerate(movies):
            if movie.ID in user.movie_ratings:
                A[i,j] = user.movie_ratings[movie.ID]
    return A

if __name__ == "__main__":
    # Read the movies and ratings
    n = 1000
    movies = read_movies("data/movies.csv", nlines="all")
    users = read_ratings("data/ratings.csv", nlines="all")
    # Shuffle the users and movies. For users necessary, to pick a random subset
    random.seed(42)
    random.shuffle(movies)
    random.shuffle(users)
    test_users = users[:10]
    users = users[10:]
    # Create the user-movie matrix
    A = create_user_movie_matrix(users, movies)
    print(f"User-movie matrix shape: {A.shape}")
    # Approximate the matrix using SVD
    k = 30
    U, S, V = np.linalg.svd(A)
    U = U[:,0:k]
    print(f"U shape: {U.shape}")
    S = np.diag(S[0:k])
    print(f"S shape: {S.shape}")
    V = V[0:k,:]
    print(f"V shape: {V.shape}")
    A_approx = U @ S @ V
    
    # test user-movie matrix
    TM = create_user_movie_matrix(test_users, movies)
    tester_ind = 0
    test0 = TM[0,:]
    plt.figure()
    plt.plot(test0)
    print(f"User 0 ratings (N={len([r for r in test0 if r > 0])} or {len(test_users[tester_ind].movie_ratings)}) ({test0.shape}): \n{test0}")
    # project the user vector to the concept space (should have 10 dimensions)
    user0_concept = test0.T @ V.T
    print(f"User 0 concept ({user0_concept.shape}): \n{user0_concept}")
    # Map the concept vector to the movie space (should have order of 1000 dimensions)
    user0_movies = user0_concept @ V
    print(f"User 0 movies approx ({user0_movies.shape}): \n{user0_movies}")
    plt.plot(user0_movies)

    # Calculate the error between the original and approximate matrix
    err = svd_approx_error(A, A_approx)
    print(f"Error: {err}")
    plt.show()