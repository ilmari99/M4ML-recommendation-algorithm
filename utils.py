import pickle
import random
from User import User
from Movie import Movie
import numpy as np
import json



def read_movies(filename, nlines="all", verbose = 0) -> list[Movie]:
    """ Reads the movies from the file and returns a list of Movie objects, where indices are sequential,
     but movie ids are not.
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
            # If the title has comma, then it is in quotes and the split is incorrect
            elif len(sp) > 3:
                ID = sp[0]
                genres = sp[-1]
                title = get_title_from_movie_data(line)
            ID = int(ID)
            genres = genres.split("|")
            movie = Movie(ID, title, genres)
            movies.append(movie)
            if verbose > 0:
                print(movie)
    return movies


def get_title_from_movie_data(line : str) -> str:
    """ Returns the title from the movie data line, if it is in quotes"""
    first_quote = line.find('"')
    last_quote = line.rfind('"')
    if first_quote == -1 or last_quote == -1:
        return line.split(",")[1]
    return line[first_quote+1:last_quote]


def read_ratings(filename, nlines = "all") -> list[User]:
    """ Reads the ratings from the file and returns a list of User objects,
    where indices are sequential, but user ids are not.
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
    """ Create a user-movie matrix from a list of users and a list of movies.
    The matrix is of size (len(users), len(movies)).
    The index (i,j) of the matrix is the rating of user i for movie j.
    If the user has not rated the movie, the value is 0.
    The ids are sequential.
    """
    # Create a matrix of zeros
    A = np.zeros((len(users), len(movies)))
    # Fill in the matrix with the user ratings
    for i, user in enumerate(users):
        for j, movie in enumerate(movies):
            if movie.ID in user.movie_ratings:
                A[i,j] = user.movie_ratings[movie.ID]
                #A[i,j] = user.movie_ratings[movieid_to_sequential_movieid[str(movie.ID)]]
    return A

def get_R_and_TM(R_file="R1.npy", TM_file="TM1.npy", save=False, test_size = 0.1):
    """ Returns the Rating (R) and test (TM) matrices.

    If R_file and TM_file are specified, the matrices are loaded from those files, if they exist.
    Both R and TM must either be loaded from file, or both must not exist.

    If R_file or TM_file are not specified, or the files don't exist, the matrices are created from the data files.
    If save is True, the matrices are saved to the files specified by R_file and TM_file.
    """
    R = None
    TM = None
    if R_file:
        try:
            R = np.load(R_file)
        except FileNotFoundError:
            R = None
            print(f"File {R_file} not found. Creating new matrix.")
    if TM_file:
        try:
            TM = np.load(TM_file)
        except FileNotFoundError:
            TM = None
            print(f"File {TM_file} not found. Creating new matrix.")
    # Both must either be from file, or both must be None
    if R is None != TM is None:
        raise ValueError("R and TM must both be None or both be not None")
    
    if R is None:
        movies = read_movies("data/movies.csv", nlines="all")
        users = read_ratings("data/ratings.csv", nlines="all")
        random.shuffle(users)
        R = create_user_movie_matrix(users, movies)
        test_sz = round(len(R[:,0])*test_size)
        test_users = users[0:test_sz]
        users = users[test_sz:]
        TM = create_user_movie_matrix(test_users, movies)
    if save:
        if not R_file:
            print("R_file not specified. Not saving R")
        if not TM_file:
            print("TM_file not specified. Not saving TM")
        np.save(R_file, R)
        np.save(TM_file, TM)
    return R, TM

def get_users_movies(movie_file = "", rating_file = "", from_pickle = True) -> tuple[list[Movie], list[User]]:
    """ Returns a tuple of the movies and users.
    Reads from either regular data files, or pkl files.
    """
    if from_pickle:
        with open("movies.pkl", "rb") as f:
            movies = pickle.load(f)
        with open("users.pkl", "rb") as f:
            users = pickle.load(f)
    else:
        movies = read_movies(movie_file, nlines="all")
        users = read_ratings(rating_file, nlines="all")
    return movies, users

def get_id_conversions(make_global = False):
    """ Reads the id conversion dictionaries from the json files and returns them.
    If make_global is True, the dictionaries are also made global.

    The dictionaries are:
        movieid_to_sequential_movieid : Which converts an actual movie id to a sequential movie id
        sequential_movieid_to_movieid : Which converts a sequential movie id to an actual movie id
        userid_to_sequential_userid : Which converts an actual user id to a sequential user id
        sequential_userid_to_userid : Which converts a sequential user id to an actual user id
    
    """
    with open("movieid_to_sequential_movieid.json", "r") as f:
        movieid_to_sequential_movieid = json.load(f)
    with open("sequential_movieid_to_movieid.json", "r") as f:
        sequential_movieid_to_movieid = json.load(f)
    with open("userid_to_sequential_userid.json", "r") as f:
        userid_to_sequential_userid = json.load(f)
    with open("sequential_userid_to_userid.json", "r") as f:
        sequential_userid_to_userid = json.load(f)
    if make_global:
        globals()["movieid_to_sequential_movieid"] = movieid_to_sequential_movieid
        globals()["sequential_movieid_to_movieid"] = sequential_movieid_to_movieid
        globals()["userid_to_sequential_userid"] = userid_to_sequential_userid
        globals()["sequential_userid_to_userid"] = sequential_userid_to_userid
    return movieid_to_sequential_movieid, sequential_movieid_to_movieid, userid_to_sequential_userid, sequential_userid_to_userid
